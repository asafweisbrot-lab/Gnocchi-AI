"""Run the demo analysis end to end and write demo_data.json for the demo page.

Steps mirror the product: processed -> discover -> classify -> map -> learn -> trace -> momentum -> prove -> find
-> conclude (decisions, open-weights readiness). Everything is computed at team level. Impact comes from comparisons
(staggered rollout, usage dose, record-level with controls), never from text similarity.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from gnocchi.analysis.purpose import CATEGORIES, classify
from gnocchi.analysis.similarity import DRIVERS, kr_targets, match
from gnocchi.synth.common import OUT, ROOT, SRC

DEMO = ROOT / "data" / "demo"
TEAMS = ["EMEA", "NA East", "NA West", "LATAM", "APAC"]


def load():
    opps = pd.concat([pd.read_csv(OUT / f"crm_opportunity_history_{y}.csv") for y in (2023, 2024)])
    ev = pd.concat([pd.read_csv(OUT / f"crm_change_log_{y}.csv") for y in (2023, 2024)])
    sess = pd.read_json(OUT / "ai_telemetry_2024.jsonl", lines=True)
    usage = pd.read_csv(OUT / "ai_admin_usage_2024.csv")
    inv = pd.read_csv(OUT / "ai_investments.csv")
    okrs = json.loads((OUT / "okrs_2024.json").read_text())
    gold = json.loads((OUT / "_gold_okr_crm_mapping.json").read_text())
    opps["amount"] = opps.amount.fillna(0).clip(lower=0)
    opps = opps.dropna(subset=["team"])
    for c in ("created_date", "close_date"):
        opps[c] = pd.to_datetime(opps[c])
    # flags derived from the change log, the way Gnocchi would see them
    se = set(ev[(ev.object == "OpportunityTeamMember") & (ev.new_value == "Solutions Engineer")].record_id)
    prep = set(ev[(ev.new_value == "Meeting prep brief")].record_id)
    sdr = set(ev[(ev.changed_by == "svc_ai_sdr")].record_id)
    notes = set(sess[sess.investment_id.isin(["inv_dealsum_emea", "inv_dealsum_nae"])].record_ids.dropna())
    opps["se"] = opps.opportunity_id.isin(se).astype(int)
    opps["prep"] = opps.opportunity_id.isin(prep).astype(int)
    opps["sdr"] = opps.opportunity_id.isin(sdr).astype(int)
    opps["dealsum"] = opps.opportunity_id.isin(notes).astype(int)
    opps["month"] = opps.created_date.dt.to_period("M").astype(str)
    opps["cmonth"] = opps.close_date.dt.to_period("M").astype(str)
    return opps, ev, sess, usage, inv, okrs, gold


def fmt_k(v) -> str:
    return f"${v/1e6:.2f}M" if v >= 1e6 else f"${round(v/1000)}K"


def ols(df: pd.DataFrame, y: str, xs: list[str], fe: list[str]):
    X = pd.get_dummies(df[xs + fe], columns=fe, drop_first=True, dtype=float)
    X.insert(0, "const", 1.0)
    A, b = X.to_numpy(float), df[y].to_numpy(float)
    coef, *_ = np.linalg.lstsq(A, b, rcond=None)
    resid = b - A @ coef
    s2 = resid @ resid / max(1, len(b) - A.shape[1])
    se = np.sqrt(np.clip(np.diag(s2 * np.linalg.pinv(A.T @ A)), 0, None))
    return {x: (float(coef[i + 1]), float(se[i + 1])) for i, x in enumerate(xs)}


def boot(df, fn, n=300, seed=7):
    g = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        idx = g.integers(0, len(df), len(df))
        vals.append(fn(df.iloc[idx]))
    vals = np.sort(vals)
    return float(np.percentile(vals, 5)), float(np.percentile(vals, 95))


def main() -> None:
    opps, ev, sess, usage, inv, okrs, gold = load()
    new = opps[opps.type == "New"].copy()
    new["log_amt"] = np.log1p(new.amount)

    # ---- 1 Discover
    discover = {
        "deals": {str(y): int((opps.created_date.dt.year.eq(y) | opps.close_date.dt.year.eq(y)).sum()) for y in (2023, 2024)},
        "deals_by_year": opps.groupby(opps.close_date.dt.year).size().astype(int).to_dict(),
        "events": int(len(ev)), "sessions": int(len(sess)), "investments": int(len(inv)),
        "teams": TEAMS,
        "inventory": [{"id": r.investment_id, "name": r["name"], "kind": r.kind, "ai_type": r.ai_type,
                       "identity": r.identity, "cost": int(r.annual_cost_usd), "teams": r.teams.split(";")}
                      for _, r in inv.iterrows()],
        "sample_change_log": ev[ev.record_id == new[(new.sdr == 1) & (new.prep == 1)].opportunity_id.iloc[0]]
            .sort_values("ts").fillna("").to_dict("records"),
        "sample_sessions": pd.concat([sess[sess.investment_id == i].head(1) for i in inv.investment_id])
            .fillna("").to_dict("records"),
    }

    # ---- 2 Map
    kr_text = {kr["kr_id"]: kr["text"] for o in okrs if o["team"] == "EMEA" for kr in o["key_results"]}
    mapping = [{"kr": kr_text[g_["kr_id"]], "record": g_["record"], "fields": g_["fields"],
                "grain": g_["time_slice"], "blind_spot": g_["record"] is None}
               for g_ in gold if g_["kr_id"].startswith("EMEA-")]

    # ---- 3 Learn drivers of the New-deal cycle (both years, controls: deal size, team, month)
    d = ols(new, "cycle_days", ["first_response_hours", "se", "prep", "meetings", "log_amt"], ["team", "month"])
    drivers = [
        {"driver": "First response time", "per": "per hour slower", "days": d["first_response_hours"]},
        {"driver": "Solutions engineer before Proposal", "per": "when added", "days": d["se"]},
        {"driver": "Meeting-prep brief attached", "per": "when attached", "days": d["prep"]},
        {"driver": "Customer meetings", "per": "per extra meeting", "days": d["meetings"]},
    ]
    drivers = [{**x, "days": round(x["days"][0], 1), "ci": round(1.645 * x["days"][1], 1)} for x in drivers]

    # ---- 4 Trace: session purpose -> similarity to drivers and key results, then triangulate
    s = sess.copy()
    targets = DRIVERS + kr_targets(okrs)
    ids, score, sim = match(s.prompt.tolist(), targets)
    nd = len(DRIVERS)
    s["match"] = ids
    s["score"] = score
    s["best_driver"] = [DRIVERS[i]["id"] if sim[k, i] >= 0.08 else None for k, i in enumerate(sim[:, :nd].argmax(1))]
    s["best_kr"] = [targets[nd + i]["id"] if sim[k, nd + i] >= 0.08 else None for k, i in enumerate(sim[:, nd:].argmax(1))]
    s["aligned"] = s.match.notna()
    s["writes_record"] = s.record_ids.notna() & s.writes.ne("none")
    # which operational element each tool's writes land on
    WRITE_ELEMENT = {"Activity (Task)": "D1", "Doc linked to opportunity": "D2"}
    sig = {"D1": abs(drivers[0]["days"]) > drivers[0]["ci"], "D2": abs(drivers[2]["days"]) > drivers[2]["ci"],
           "D3": abs(drivers[1]["days"]) > drivers[1]["ci"], "D5": abs(drivers[3]["days"]) > drivers[3]["ci"], "D4": False}
    tnames = {t["id"]: t["name"] for t in targets}
    sim_matrix, trace = [], []
    for i in inv.investment_id:
        x = s[s.investment_id == i]
        shares = {t["id"]: round(float((x.match == t["id"]).mean()), 3) for t in targets}
        shares["none"] = round(float(x.match.isna().mean()), 3)
        sim_matrix.append({"id": i, "n": int(len(x)), "shares": shares})
        written = [WRITE_ELEMENT.get(w) for w in x[x.writes_record].writes.unique()]
        written_drivers = [w for w in written if w]
        top = x.best_driver.value_counts(normalize=True)
        top_d = top.index[0] if len(top) else None
        agrees = bool(written_drivers) and top_d in written_drivers
        al = float(x.aligned.mean())
        if (x.writes == "Commit to repo").any():
            level = "2 · Via what it built"
            why = (f"Writes code, not CRM records: its sessions commit to the repos that deploy the deal-summary bots, "
                   f"so its impact is theirs (step 7)")
        elif written_drivers and any(sig.get(w) for w in written_drivers) and agrees:
            level = "1 · Closed triangle"
            why = (f"Writes {tnames[written_drivers[0]].lower()} records; {round(100*top.get(top_d,0))}% of its sessions are aimed "
                   f"at that driver, which moves the cycle key result")
        elif x.writes_record.any():
            level = "2 · Records only"
            why = (f"Writes notes on opportunities, an element no driver depends on; "
                   f"{round(100*float(x.match.isna().mean()))}% of its sessions match no driver or key result")
        elif al >= 0.2:
            level = "3 · OKR only / unaligned"
            why = f"No record trace; {round(100*al)}% of sessions match a driver or key result, the rest serve no stated goal"
        else:
            level = "Unaligned"
            why = f"No record trace; only {round(100*al)}% of sessions match a driver or key result"
        trace.append({"id": i, "level": level, "why": why, "aligned": round(al, 3)})
    # agreement with the planted purposes (synthetic ground truth only)
    gt = pd.read_csv(OUT / "_ground_truth_session_purpose.csv").set_index("session_id").purpose
    EXPECT = {"Outreach: first touch": {"D1", "KR2"}, "Research: account and contact": {"D1", "KR2"},
              "Research: meeting prep brief": {"D2"}, "Research: account background": {"D2", "D1", None},
              "Drafting: proposal text": {"D4"}, "Summarize: deal notes": {None, "D5"},
              "Drafting: internal email": {None}, "Summarize: meeting notes": {None, "D2", "D5"}, "Personal productivity": {None},
              "Analysis: pipeline and cycle": {None, "KR3", "KR1", "KR2"}, "Strategy: account and territory plans": {None, "KR1", "KR2", "D4"},
              "Development: formulas and scripts": {None}, "Development: build deal-summary bot": {None, "D5", "D2"}}
    s["truth"] = s.session_id.map(gt)
    s["ok"] = [(None if pd.isna(m) else m) in EXPECT[t] for m, t in zip(s.match, s.truth)]
    agree = {p: round(float(g_.ok.mean()), 3) for p, g_ in s.groupby("truth")}
    examples = []
    for i in inv.investment_id:
        xi = s[s.investment_id == i]
        m_ = xi[xi.aligned]
        r = m_.sort_values("score").iloc[len(m_) // 2] if len(m_) else xi.iloc[0]
        k = s.index.get_loc(r.name)
        order = np.argsort(-sim[k])[:3]
        examples.append({"id": i, "prompt": r.prompt, "writes": r.writes,
                         "top": [{"target": targets[j]["name"], "kind": "driver" if j < nd else "key result",
                                  "score": round(float(sim[k, j]), 2)} for j in order]})
    similarity = {"targets": [{"id": t["id"], "name": t["name"], "kind": "driver" if t["id"].startswith("D") else "key result"} for t in targets],
                  "matrix": sim_matrix, "examples": examples, "agreement": agree,
                  "overall_agreement": round(float(s.ok.mean()), 3), "threshold": 0.08}
    chat = s[s.investment_id == "inv_chatgpt_ent"]

    # ---- 2 Classify: what is the AI used for, and is it direct or indirect activity
    cat, _ = classify(s.prompt.tolist())
    s["category"] = cat
    # metadata beats text: a session that commits to a repo is development work, even when its
    # text is about summarizing notes
    s["category_text"] = cat
    s.loc[s.writes == "Commit to repo", "category"] = "Development"
    s["mode"] = np.where(s.identity.ne("sso_user"), "direct",
                         np.where(s.writes_record, "indirect_traced", "indirect_untraced"))
    cats = [c["id"] for c in CATEGORIES]
    gtc = pd.read_csv(OUT / "_ground_truth_session_purpose.csv").set_index("session_id").category
    by_inv = []
    for i in inv.investment_id:
        x = s[s.investment_id == i]
        by_inv.append({"id": i, "n": int(len(x)), "mode": x["mode"].mode().iloc[0],
                       "shares": {c: round(float((x.category == c).mean()), 3) for c in cats}})
    classify_out = {
        "categories": cats, "by_investment": by_inv,
        "by_mode": {m: {c: int(((s["mode"] == m) & (s.category == c)).sum()) for c in cats}
                    for m in ("direct", "indirect_traced", "indirect_untraced")},
        "tokens": {c: int(s[s.category == c].tokens.sum()) for c in cats},
        "aimed": {c: round(float(s[s.category == c].aligned.mean()), 3) for c in cats},
        "agreement": round(float((s.category == s.session_id.map(gtc)).mean()), 3),
        "agreement_text_only": round(float((s.category_text == s.session_id.map(gtc)).mean()), 3),
        "examples": {c: s[s.category == c].prompt.iloc[0] for c in cats if (s.category == c).any()},
    }

    # ---- 1b Build lineage: coding sessions -> repo -> deployed identity -> CRM records
    builds = pd.read_csv(OUT / "code_build_log.csv")
    by_ident = inv.set_index("identity")
    lineage = []
    code = s[s.writes == "Commit to repo"]
    for _, b in builds.iterrows():
        bot = by_ident.loc[b.deploys_identity]
        cs = code[code.record_ids == b.repo]
        pre = cs[cs.ts < b.deployed_at]
        written = s[(s.identity == b.deploys_identity) & s.writes_record]
        lineage.append({"repo": b.repo, "team": b.team, "bot_id": bot.investment_id, "bot": bot["name"],
                        "identity": b.deploys_identity, "sessions": int(len(cs)), "build_sessions": int(len(pre)),
                        "first_commit": b.first_commit, "deployed_at": b.deployed_at, "commits": int(b.commits),
                        "engineer_days": int(b.engineer_days), "build_cost": int(b.engineer_days * 700),
                        "records": int(written.record_ids.nunique()), "run_cost": int(bot.annual_cost_usd)})
    # same bot built twice? compare the two builds' coding sessions
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    a = code[code.record_ids == builds.repo[0]].prompt.tolist()
    b_ = code[code.record_ids == builds.repo[1]].prompt.tolist()
    other = s[(s.investment_id == "inv_chatgpt_ent")].prompt.tolist()
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit(a + b_ + other)
    best = cosine_similarity(vec.transform(b_), vec.transform(a)).max(1)
    base = cosine_similarity(vec.transform(b_), vec.transform(other)).max(1)
    duplicate = {"later": builds.repo[1], "earlier": builds.repo[0], "share_near_identical": round(float((best >= 0.35).mean()), 3),
                 "mean_best": round(float(best.mean()), 2), "mean_baseline": round(float(base.mean()), 2),
                 "wasted_cost": int(builds.engineer_days[1] * 700)}

    # ---- 5 Momentum (monthly, 2024)
    s["m"] = pd.to_datetime(s.ts).dt.month
    mom = {}
    for i in inv.investment_id:
        u = usage[usage.investment_id == i].copy()
        u["m"] = u.month.str[-2:].astype(int)
        uu = u.groupby("m").agg(assigned=("licences_assigned", "sum"), active=("active_users", "sum"), sessions=("sessions", "sum"))
        recs = s[(s.investment_id == i) & s.record_ids.notna()].groupby("m").record_ids.nunique()
        aimed = s[s.investment_id == i].groupby("m").aligned.mean()
        mom[i] = [{"m": int(m), "sessions": int(uu.loc[m, "sessions"]) if m in uu.index else 0,
                   "active_share": round(float(uu.loc[m, "active"] / uu.loc[m, "assigned"]), 2) if m in uu.index and uu.loc[m, "assigned"] else 0,
                   "records": int(recs.get(m, 0)), "aimed": round(float(aimed.get(m, 0)), 2)} for m in range(1, 13)]
    n24 = new[new.close_date.dt.year == 2024].copy()
    n24["m"] = n24.close_date.dt.month
    driver_trend = {
        "resp_emea": [round(float(n24[(n24.team == "EMEA") & (n24.m == m)].first_response_hours.median()), 1) for m in range(1, 13)],
        "resp_others": [round(float(n24[(~n24.team.isin(["EMEA", "NA East"])) & (n24.m == m)].first_response_hours.median()), 1) for m in range(1, 13)],
    }

    # ---- 6 Prove
    # AI SDR: staggered rollout, team and month fixed effects, deal size control
    def sdr_eff(df):
        return ols(df, "cycle_days", ["sdr", "prep", "se", "log_amt"], ["team", "cmonth"])["sdr"][0]
    sdr_est = sdr_eff(new)
    sdr_lo, sdr_hi = boot(new, sdr_eff)
    def resp_eff(df):
        return ols(df, "first_response_hours", ["sdr"], ["team", "cmonth"])["sdr"][0]
    resp_est = resp_eff(new)
    # Meeting-prep: raw record-level vs adjusted for team, month, SDR, SE
    ne = opps[opps.type.isin(["New", "Expansion"]) & (opps.close_date >= "2024-07-01")].copy()
    ne["log_amt"] = np.log1p(ne.amount)
    raw_prep = float(ne[ne.prep == 1].cycle_days.mean() - ne[ne.prep == 0].cycle_days.mean())
    def prep_eff(df):
        return ols(df, "cycle_days", ["prep", "sdr", "se", "log_amt"], ["team", "month", "type"])["prep"][0]
    prep_est = prep_eff(ne)
    prep_lo, prep_hi = boot(ne, prep_eff)
    dose = []
    for t in TEAMS:
        h2 = ne[ne.team == t]
        dose.append({"team": t, "prep_share": round(100 * float(h2.prep.mean()), 0) if len(h2) else 0})
    # Deal-summary bots: any cycle effect on the deals they touched?
    ds = opps[opps.type.isin(["New", "Expansion", "Renewal"]) & (opps.close_date >= "2024-03-01") & opps.team.isin(["EMEA", "NA East"])].copy()
    ds["log_amt"] = np.log1p(ds.amount)
    def ds_eff(df):
        return ols(df, "cycle_days", ["dealsum", "sdr", "prep", "se", "log_amt"], ["team", "month", "type"])["dealsum"][0]
    ds_est = ds_eff(ds)
    ds_lo, ds_hi = boot(ds, ds_eff)
    # KR3 view: median New-deal cycle by team and year
    kr3 = (new.assign(y=new.close_date.dt.year).groupby(["team", "y"]).cycle_days.median().unstack()).round(0)
    kr3_rows = [{"team": t, "y2023": int(kr3.loc[t, 2023]), "y2024": int(kr3.loc[t, 2024])} for t in TEAMS]
    # monthly cycle, treated vs comparison teams
    m_series = []
    for grp, teams in [("EMEA (AI SDR from Apr)", ["EMEA"]), ("NA East (from Jul)", ["NA East"]), ("Teams without AI SDR", ["NA West", "LATAM", "APAC"])]:
        x = new[new.team.isin(teams) & (new.close_date.dt.year == 2024)]
        q = x.close_date.dt.quarter
        m_series.append({"group": grp, "values": [round(float(x[q == k].cycle_days.median()), 1) for k in range(1, 5)]})
    # Team fit: which share of each team's AI sessions (and spend) is aimed at a driver or KR,
    # set against the team's KPIs. A pattern to explain, not proof: the comparisons above prove.
    cost = inv.set_index("investment_id").annual_cost_usd
    resp_y = new.assign(y=new.close_date.dt.year).groupby(["team", "y"]).first_response_hours.median().unstack()
    nb = new.assign(y=new.close_date.dt.year).groupby(["team", "y"]).amount.sum().unstack()
    team_fit = []
    for t in TEAMS:
        xt = s[s.team == t]
        spend = aimed_spend = 0.0
        tools = []
        for i in inv.investment_id:
            xi, xall = xt[xt.investment_id == i], s[s.investment_id == i]
            if not len(xi):
                continue
            c = float(cost[i]) * len(xi) / len(xall)
            spend += c
            aimed_spend += c * float(xi.aligned.mean())
            top = xi.match.value_counts(normalize=True)
            tools.append({"id": i, "n": int(len(xi)), "aimed": round(float(xi.aligned.mean()), 3),
                          "top": top.index[0] if len(top) else None, "spend": round(c)})
        sig_ids = [k for k, v in sig.items() if v]
        team_fit.append({
            "team": t, "sessions": int(len(xt)), "aimed": round(float(xt.aligned.mean()), 3),
            "on_driver": round(float(xt.match.isin(sig_ids).mean()), 3),
            "spend": round(spend), "aimed_spend": round(aimed_spend), "tools": tools,
            "kpi": {"cycle": [int(kr3.loc[t, 2023]), int(kr3.loc[t, 2024])],
                    "resp": [round(float(resp_y.loc[t, 2023]), 1), round(float(resp_y.loc[t, 2024]), 1)],
                    "new_biz": [round(float(nb.loc[t, 2023])), round(float(nb.loc[t, 2024]))]}})
    prove = {
        "team_fit": team_fit,
        "sdr": {"cycle": round(sdr_est, 1), "lo": round(sdr_lo, 1), "hi": round(sdr_hi, 1),
                "resp": round(resp_est, 1), "deals": int(new.sdr.sum()), "series": m_series},
        "prep": {"raw": round(raw_prep, 1), "adjusted": round(prep_est, 1), "lo": round(prep_lo, 1), "hi": round(prep_hi, 1),
                 "briefs": int(ne.prep.sum()), "dose": dose},
        "dealsum": {"cycle": round(ds_est, 1), "lo": round(ds_lo, 1), "hi": round(ds_hi, 1), "deals": int(ds.dealsum.sum())},
        "kr3": kr3_rows,
    }

    # ---- 7 Find: AI opportunities and blind spots
    se_share = new.groupby("team").se.mean()
    best = float(se_share.max())
    se_w = -d["se"][0]
    find = [{"team": t, "se_share": round(100 * float(se_share[t])), "gap_pp": round(100 * (best - float(se_share[t]))),
             "days_at_stake": round(se_w * (best - float(se_share[t])), 1)} for t in TEAMS]

    # ---- 8 Report
    chat_mom = mom["inv_chatgpt_ent"]
    act_first, act_last = chat_mom[0]["active_share"], chat_mom[-1]["active_share"]
    seat_cut = 0.4
    report = [
        {"id": "inv_ai_sdr", "name": "AI SDR agent", "cost": 96000, "level": "Closed triangle",
         "impact": f"{prove['sdr']['cycle']} days on New-deal cycle (90% range {prove['sdr']['lo']} to {prove['sdr']['hi']}); first response {prove['sdr']['resp']} h",
         "method": "Difference-in-differences: staggered rollout (EMEA Apr, NA East Jul) vs teams without it, team and close-month fixed effects", "confidence": "High",
         "decision": "Renew, then expand to NA West and LATAM in waves", "avoided": 0},
        {"id": "inv_meeting_prep", "name": "Meeting-prep assistant", "cost": 38000, "level": "Closed triangle",
         "impact": f"{prove['prep']['adjusted']} days per brief after controls (raw gap {prove['prep']['raw']} days overstates it)",
         "method": "Record-level with team, month, SDR and SE controls; usage dose by team", "confidence": "Medium",
         "decision": "Renew; lift adoption in APAC and LATAM", "avoided": 0},
        {"id": "inv_dealsum", "name": "Deal-summary bots (EMEA and NA East builds)", "cost": 40000, "level": "Records only",
         "impact": f"{prove['dealsum']['cycle']} days (90% range {prove['dealsum']['lo']} to {prove['dealsum']['hi']}): no measurable effect",
         "method": "Record-level with controls", "confidence": "Medium",
         "decision": "Consolidate into one build", "avoided": 18000,
         "note": f"Built twice: {sum(l['engineer_days'] for l in lineage)} engineer-days (${sum(l['build_cost'] for l in lineage)//1000}K) of build effort on top of run cost"},
        {"id": "inv_coding_asst", "name": "Coding assistant (RevOps builders)", "cost": 9600, "level": "Via what it built",
         "impact": f"Built both deal-summary bots ({sum(l['sessions'] for l in lineage)} sessions, {int(builds.commits.sum())} commits); "
                   f"the second build's sessions match the first's {duplicate['mean_best']/max(duplicate['mean_baseline'],0.01):.0f}× more closely than other work; its impact is the bots': none measured",
         "method": "Lineage: coding sessions -> repo -> CI deploy of the bot identity -> CRM records", "confidence": "Medium",
         "decision": "Keep for builders; register each build so a duplicate is caught before it starts", "avoided": 0},
        {"id": "inv_chatgpt_ent", "name": "ChatGPT Enterprise (sales seats)", "cost": 54000, "level": next(t["level"] for t in trace if t["id"] == "inv_chatgpt_ent").split("· ")[-1],
         "impact": f"Active seats fell from {round(100*act_first)}% to {round(100*act_last)}%; {100-round(100*chat.aligned.mean())}% of sessions serve no stated goal",
         "method": "Momentum and purpose only; no record trace to compare", "confidence": "Low",
         "decision": "Right-size seats at renewal", "avoided": int(54000 * seat_cut)},
    ]
    # ---- 9 Open-weights readiness (the customer's): which workloads could move to an open model
    proven = {"inv_ai_sdr": f"{prove['sdr']['cycle']} days (rollout comparison)",
              "inv_meeting_prep": f"{prove['prep']['adjusted']} days (record level)"}
    cost = inv.set_index("investment_id").annual_cost_usd
    REASONING = {"Analysis", "Strategy", "Development"}
    workloads = []
    for (i, c), x in s.groupby(["investment_id", "category"]):
        if len(x) < 20:
            continue
        n_inv = int((s.investment_id == i).sum())
        spend = float(cost[i]) * len(x) / n_inv
        if c in REASONING:
            verdict, why = "Keep on frontier model", "Open-ended reasoning or code; low volume"
        elif i in proven:
            verdict, why = "Ready to pilot", f"Routine, repeatable; proven effect {proven[i]} to re-test a swap against"
        elif i.startswith("inv_dealsum"):
            verdict, why = "Consolidate, then move", "Routine; no effect to lose, one build instead of two"
        else:
            verdict, why = "Not ready", "Routine, but no record outcome to check a swap against"
        workloads.append({"id": i, "category": c, "sessions": int(len(x)), "tokens": int(x.tokens.sum()),
                          "mode": x["mode"].iloc[0], "spend": round(spend), "verdict": verdict, "why": why})
    order = {"Ready to pilot": 0, "Consolidate, then move": 1, "Not ready": 2, "Keep on frontier model": 3}
    workloads.sort(key=lambda w: (order[w["verdict"]], -w["spend"]))
    # after consolidation only one deal-summary build remains, so the retired one is not in scope
    in_scope = sum(w["spend"] for w in workloads if w["verdict"] in ("Ready to pilot", "Consolidate, then move")
                   and w["id"] != "inv_dealsum_emea")
    CUT = (0.4, 0.7)  # assumption: run-cost cut for a comparable open model hosted by the customer
    traced = s["mode"].isin(["direct", "indirect_traced"])
    open_weights = {
        "workloads": workloads, "in_scope": round(in_scope), "cut": CUT,
        "potential": [round(in_scope * CUT[0]), round(in_scope * CUT[1])],
        "readiness": [
            {"item": "Workloads classified by purpose", "status": "done", "detail": f"{len(s):,} sessions in {len(cats)} categories"},
            {"item": "Outcome baseline to re-test a swap", "status": "partial",
             "detail": f"{fmt_k(sum(cost[i] for i in proven))} of {fmt_k(cost.sum())} spend has a proven effect"},
            {"item": "Record-linked sessions (evaluation set)", "status": "partial",
             "detail": f"{int(traced.sum()):,} sessions ({round(100*traced.mean())}%) are tied to CRM records"},
            {"item": "Hosting and data policy for an open model", "status": "open", "detail": "Customer decision; not measured here"},
        ],
    }

    # ---- 0 Processed: what this run read
    def rows_xlsx(f):
        return int(len(pd.read_excel(SRC / f)))
    processed = [
        {"source": "CRM closed-won deals, 2023–2024", "kind": "real", "rows": int(len(opps)), "unit": "deals"},
        {"source": "Sales targets by owner", "kind": "real", "rows": rows_xlsx("personal_targets.xlsx"), "unit": "rows"},
        {"source": "KPI definitions and CRM field map", "kind": "real",
         "rows": len(json.loads((SRC / "kpi_definitions.json").read_text())["targets"]), "unit": "KPIs"},
        {"source": "Team OKRs 2024", "kind": "syn", "rows": sum(len(o["key_results"]) for o in okrs), "unit": "key results"},
        {"source": "CRM change history", "kind": "syn", "rows": int(len(ev)), "unit": "events"},
        {"source": "AI telemetry (session text)", "kind": "syn", "rows": int(len(sess)), "unit": "sessions"},
        {"source": "AI admin and usage reports", "kind": "syn", "rows": int(len(usage)), "unit": "rows"},
        {"source": "AI investment register", "kind": "syn", "rows": int(len(inv)), "unit": "investments"},
        {"source": "Code repos and CI deploy log", "kind": "syn", "rows": int(builds.commits.sum()), "unit": "commits"},
    ]
    pipeline = [
        ["Discover", f"{len(inv)} AI investments", f"{int((inv.ai_type == 'acting').sum())} direct, {int((inv.ai_type != 'acting').sum())} indirect, {len(lineage)} builds"],
        ["Classify", f"{len(s):,} sessions", f"{len(cats)} work categories"],
        ["Map", f"{sum(len(o['key_results']) for o in okrs)} key results", f"{sum(1 for g_ in gold if g_['record'] is None)} blind spots"],
        ["Learn", f"{int(len(new)):,} new deals", f"{sum(1 for x in drivers if abs(x['days']) > x['ci'])} drivers that move the cycle"],
        ["Trace", f"{len(targets)} drivers and KRs", f"{sum(1 for t in trace if t['level'].startswith('1'))} closed triangles"],
        ["Momentum", "12 months", "1 tool fading"],
        ["Prove", "3 comparisons", "2 effects, 1 null"],
        ["Find", "5 teams", "1 AI opportunity"],
        ["Conclude", f"{fmt_k(inv.annual_cost_usd.sum())} AI spend", f"{len(report)} decisions"],
    ]
    out = {"lineage": lineage, "duplicate": duplicate, "processed": processed, "pipeline": pipeline, "classify": classify_out, "open_weights": open_weights,
           "discover": discover, "map": mapping, "drivers": drivers, "trace": trace, "similarity": similarity, "momentum": mom,
           "driver_trend": driver_trend, "prove": prove, "find": find, "report": report,
           "totals": {"ai_spend": int(inv.annual_cost_usd.sum()), "avoided": sum(r["avoided"] for r in report)}}
    DEMO.mkdir(parents=True, exist_ok=True)
    (DEMO / "demo_data.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({"drivers": drivers, "sdr": prove["sdr"]["cycle"], "sdr_range": [prove["sdr"]["lo"], prove["sdr"]["hi"]],
                      "prep": prove["prep"], "dealsum": prove["dealsum"], "find": find, "kr3": kr3_rows,
                      "chat": [act_first, act_last], "totals": out["totals"]}, indent=1, default=str))


if __name__ == "__main__":
    main()
