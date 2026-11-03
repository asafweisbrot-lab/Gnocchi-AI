"""Lab: better ways to match a session to the deal it is about.

Compares, on ChatGPT sessions with known (planted) deal links and on decoys:
  1. blob     one text per deal: account, deal type, vertical (what the report uses)
  2. blob+OKR the same plus the key results the deal counts toward, in one sentence
  3. chunks   several short texts per deal (account; type and vertical; sub-vertical and
              market; tech partner; each stage and activity; the key results), and the
              session takes its best chunk, so no field drowns another
  4. story    one short narrative per deal built from its CRM history
  5. name + judge  exact account-name match among the team's open deals, then a small
              model answers yes/no per candidate deal, reading its story

    PYTHONPATH=src python -m gnocchi.analysis.deal_match_lab
    GNOCCHI_JUDGE_MODEL=openai/qwen2.5-1.5b GNOCCHI_JUDGE_API_BASE=http://127.0.0.1:8000/v1 \\
        PYTHONPATH=src python -m gnocchi.analysis.deal_match_lab --judge
"""
from __future__ import annotations

import json
import os
import re
import statistics
import sys
import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from gnocchi.synth.common import OUT, ROOT, load_crm

DEMO = ROOT / "data" / "demo"
KIND = {"New": "new-logo new", "Expansion": "expansion", "Upsell": "upsell"}
BAND = [(25_000, "small deal under $25K"), (100_000, "mid-size deal $25K-$100K"), (1e12, "large deal over $100K")]


def load():
    s = pd.read_json(OUT / "ai_telemetry_2025.jsonl", lines=True)
    gt = pd.read_csv(OUT / "_ground_truth_session_purpose.csv").set_index("session_id")
    s = s.join(gt, on="session_id")
    s = s[(s.investment_id == "inv_chatgpt_ent") & ~s.category.isin(["Development", "Tool building", "Tool maintenance"])].copy()
    o = pd.read_csv(OUT / "crm_opportunity_history_2025.csv")
    for c in ("created_date", "close_date"):
        o[c] = pd.to_datetime(o[c])
    o = o[o.type.isin(KIND)].copy()
    crm = load_crm(2025).set_index("opportunity id")
    ev = pd.read_csv(OUT / "crm_change_log_2025.csv")
    okrs = json.loads((OUT / "okrs_2025.json").read_text())
    krt = {(x["team"], i): kr["text"] for x in okrs for i, kr in enumerate(x["key_results"], 1)}
    return s, o, crm, ev, krt


def deal_docs(o, crm, ev, krt):
    evs = ev.groupby("record_id")
    docs = {}
    for r in o.itertuples():
        c = crm.loc[r.opportunity_id]
        band = next(b for lim, b in BAND if r.amount < lim)
        krs = [krt[(r.team, 1)]] + ([krt[(r.team, 2)], krt[(r.team, 3)]] if r.type == "New" else [])
        steps = []
        if r.opportunity_id in evs.groups:
            for e in evs.get_group(r.opportunity_id).sort_values("ts").itertuples():
                if e.field == "StageName":
                    steps.append(f"{e.new_value} on {e.ts[:10]}")
                elif e.new_value in ("Proposal", "Meeting prep brief", "Solutions Engineer"):
                    steps.append(f"{e.new_value.lower()} added {e.ts[:10]}")
        chunks = [c["account name"], f"{KIND[r.type]} {c['verticals']}",
                  f"{c['sub-vertical']} {c['market']}" if isinstance(c["sub-vertical"], str) else str(c["market"]),
                  f"tech partner {c['tech partner']}" if isinstance(c["tech partner"], str) else "",
                  band, " ".join(steps), " ".join(krs)]
        story = (f"{c['account name']}: a {KIND[r.type].split()[0]} deal in {c['verticals']}, {band}, opened {r.created_date.date()}"
                 f"{', ' + ', '.join(steps[1:4]) if len(steps) > 1 else ''}, closed won {r.close_date.date()}. "
                 f"Counts toward: {krs[0]}" + (f"; {krs[1]}" if len(krs) > 1 else "") + ".")
        docs[r.opportunity_id] = {"blob": f"{c['account name']} {KIND[r.type]} {c['verticals']}",
                                  "blob_okr": f"{c['account name']} {KIND[r.type]} {c['verticals']}. Counts toward {' '.join(krs)}",
                                  "chunks": [x for x in chunks if x], "story": story, "account": c["account name"]}
    return docs


def open_mask(st, rt):
    tsv = pd.to_datetime(st.ts).values[:, None]
    return (tsv >= rt.created_date.values[None, :]) & (tsv <= rt.close_date.values[None, :])


def match_text(s, o, docs, field, thr):
    texts = [docs[i][field] for i in o.opportunity_id]
    v = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english").fit(texts + list(s.prompt))
    pred = pd.Series(None, index=s.index, dtype=object)
    for t in o.team.unique():
        st, rt = s[s.team == t], o[o.team == t]
        if not len(st):
            continue
        m = np.where(open_mask(st, rt), cosine_similarity(v.transform(st.prompt), v.transform([docs[i][field] for i in rt.opportunity_id])), 0)
        ok = m.max(1) >= thr
        pred[st.index[ok]] = rt.opportunity_id.values[m.argmax(1)[ok]]
    return pred


def match_chunks(s, o, docs, thr):
    allc = [(i, ch) for i in o.opportunity_id for ch in docs[i]["chunks"]]
    v = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english").fit([c for _, c in allc] + list(s.prompt))
    pred = pd.Series(None, index=s.index, dtype=object)
    for t in o.team.unique():
        st, rt = s[s.team == t], o[o.team == t]
        if not len(st):
            continue
        S = v.transform(st.prompt)
        # account chunk counts in full, every other chunk at half weight, best of each
        score = np.zeros((len(st), len(rt)))
        for j, i in enumerate(rt.opportunity_id):
            ch = docs[i]["chunks"]
            sm = cosine_similarity(S, v.transform(ch))
            score[:, j] = sm[:, 0] + 0.5 * (sm[:, 1:].max(1) if sm.shape[1] > 1 else 0)
        m = np.where(open_mask(st, rt), score, 0)
        ok = m.max(1) >= thr
        pred[st.index[ok]] = rt.opportunity_id.values[m.argmax(1)[ok]]
    return pred


def candidates_by_name(s, o, docs):
    """Exact account-name match among the team's deals open when the session ran."""
    out = {}
    for t in o.team.unique():
        st, rt = s[s.team == t], o[o.team == t]
        om = open_mask(st, rt)
        for a, (k, r) in enumerate(st.iterrows()):
            p = r.prompt.lower()
            out[k] = [rt.opportunity_id.values[j] for j in np.where(om[a])[0]
                      if re.search(r"\b" + re.escape(docs[rt.opportunity_id.values[j]]["account"].lower()) + r"\b", p)]
    return out


def score(pred, truth):
    made, has = pred.notna(), truth.notna()
    return {"linked": int(made.sum()), "precision": round(float((pred[made] == truth[made]).mean()), 3) if made.any() else None,
            "recall": round(float((pred[has] == truth[has]).mean()), 3)}


YESNO = "You check whether a work session is about a given sales deal. Reply with one word: yes or no."


def ask(model, session, story, when):
    import litellm
    extra = {"api_base": os.environ["GNOCCHI_JUDGE_API_BASE"], "api_key": "local"} if os.environ.get("GNOCCHI_JUDGE_API_BASE") else {}
    r = litellm.completion(model=model, temperature=0, max_tokens=3, **extra,
                           messages=[{"role": "system", "content": YESNO},
                                     {"role": "user", "content": f"Deal: {story}\nSession ({when}): {session}\nDoes the session mention this deal's account and fit this deal? Answer yes or no."}])
    return r.choices[0].message.content.strip().lower().startswith("y")


def main() -> None:
    s, o, crm, ev, krt = load()
    docs = deal_docs(o, crm, ev, krt)
    truth = s.opp
    res = {}
    for name, fn in [("Deal text (account, type, vertical)", lambda: match_text(s, o, docs, "blob", 0.3)),
                     ("Deal text + OKR in one sentence", lambda: match_text(s, o, docs, "blob_okr", 0.2)),
                     ("Chunks, best chunk per field", lambda: match_chunks(s, o, docs, 0.3)),
                     ("Story per deal", lambda: match_text(s, o, docs, "story", 0.15))]:
        res[name] = score(fn(), truth)
    cand = candidates_by_name(s, o, docs)
    # name match alone: pick the only candidate, or the one whose type the session names
    def pick_by_type(k, cs):
        if len(cs) <= 1:
            return cs[0] if cs else None
        p = s.loc[k].prompt.lower()
        typed = [c for c in cs if any(w in p for w in KIND[o.set_index("opportunity_id").loc[c, "type"]].split())]
        return (typed or cs)[0]
    name_pred = pd.Series({k: pick_by_type(k, cs) for k, cs in cand.items()}, dtype=object).reindex(s.index)
    res["Exact account name, then deal type"] = score(name_pred, truth)
    # stage at session time: a deal still before its proposal is the one being worked on
    prop = ev[ev.new_value == "Proposal"].groupby("record_id").ts.min()
    def pick_by_stage(k, cs):
        if len(cs) <= 1:
            return cs[0] if cs else None
        ts = s.loc[k].ts
        live = [c for c in cs if prop.get(c, "9999") > ts]
        p = s.loc[k].prompt.lower()
        typed = [c for c in (live or cs) if any(w in p for w in KIND[o.set_index("opportunity_id").loc[c, "type"]].split())]
        return (typed or live or cs)[0]
    stage_pred = pd.Series({k: pick_by_stage(k, cs) for k, cs in cand.items()}, dtype=object).reindex(s.index)
    res["Exact account name, then type and stage at session time"] = score(stage_pred, truth)
    named = s.index[[bool(re.search(re.escape(docs[d]["account"].lower()), s.loc[k].prompt.lower())) if isinstance(d, str) else False
                     for k, d in zip(s.index, truth)]]
    upper = round(len(named) / max(1, int(truth.notna().sum())), 3)
    out = {"variants": res, "ceiling_recall": upper, "n_sessions": int(len(s)), "true_links": int(truth.notna().sum()),
           "multi_candidate": int(sum(1 for c in cand.values() if len(c) > 1)),
           "example_story": docs[truth.dropna().iloc[0]]["story"], "example_chunks": docs[truth.dropna().iloc[0]]["chunks"]}

    if "--judge" in sys.argv:
        model = os.environ["GNOCCHI_JUDGE_MODEL"]
        rng = np.random.default_rng(7)
        pos = s[truth.notna()].index.to_numpy()
        neg = s[truth.isna() & s.index.isin([k for k, c in cand.items() if not c])].index.to_numpy()
        named_neg = s[truth.isna() & s.index.isin([k for k, c in cand.items() if c])].index.to_numpy()
        pick = list(rng.choice(pos, 45, replace=False)) + list(rng.choice(neg, 10, replace=False)) + list(named_neg[:5])
        jp, lat, calls = {}, [], 0
        for k in pick:
            cs = cand.get(k, [])[:3]
            chosen = None
            for c in cs:
                t0 = time.time()
                calls += 1
                if ask(model, s.loc[k].prompt, docs[c]["story"], s.loc[k].ts[:10]):
                    chosen = chosen or c
                lat.append(time.time() - t0)
            jp[k] = chosen
        jp = pd.Series(jp, dtype=object)
        sub_truth = truth.loc[jp.index]
        out["judge"] = {"model": model, "n": len(jp), "calls": calls,
                        "median_call_s": round(statistics.median(lat), 2) if lat else None,
                        "name_then_judge": score(jp, sub_truth),
                        "name_then_type_same_sample": score(name_pred.loc[jp.index], sub_truth),
                        "name_then_stage_same_sample": score(stage_pred.loc[jp.index], sub_truth),
                        "blob_same_sample": score(match_text(s, o, docs, "blob", 0.3).loc[jp.index], sub_truth),
                        "ran_at": time.strftime("%Y-%m-%d %H:%M")}
    elif (DEMO / "deal_match_lab.json").exists():
        prev = json.loads((DEMO / "deal_match_lab.json").read_text())
        if "judge" in prev:
            out["judge"] = prev["judge"]
    (DEMO / "deal_match_lab.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({k: v for k, v in out.items() if k != "example_chunks"}, indent=1, default=str))


if __name__ == "__main__":
    main()
