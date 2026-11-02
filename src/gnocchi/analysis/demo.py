"""Run the demo analysis end to end and write demo_data.json for the demo page.

Steps mirror the product: discover -> map -> learn -> trace -> momentum -> prove -> find
-> report. Everything is computed at team level. Impact comes from comparisons
(staggered rollout, usage dose, record-level with controls), never from text similarity.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from gnocchi.synth.common import OUT, ROOT

DEMO = ROOT / "data" / "demo"
TEAMS = ["EMEA", "NA East", "NA West", "LATAM", "APAC"]
ALIGNED_PURPOSES = {"Research: account background", "Drafting: proposal text"}


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

    # ---- 4 Trace: evidence level per investment
    s = sess.copy()
    s["aligned"] = s.purpose.isin(ALIGNED_PURPOSES)
    chat = s[s.investment_id == "inv_chatgpt_ent"]
    trace = [
        {"id": "inv_ai_sdr", "level": "1 · Closed triangle", "why": "Writes first-touch activities on New deals; first response drives the cycle key result"},
        {"id": "inv_meeting_prep", "level": "1 · Closed triangle", "why": "Prep briefs linked to opportunities; the brief is a driver of the cycle key result"},
        {"id": "inv_dealsum_emea", "level": "2 · Records only", "why": "Writes notes on opportunities, but notes don't move any driver"},
        {"id": "inv_dealsum_nae", "level": "2 · Records only", "why": "Same job as the EMEA build, in NA East: overlap"},
        {"id": "inv_chatgpt_ent", "level": "3 · OKR only / unaligned",
         "why": f"No record trace; {round(100*chat.aligned.mean())}% of sessions match an OKR purpose, the rest serve no stated goal"},
    ]

    # ---- 5 Momentum (monthly, 2024)
    s["m"] = pd.to_datetime(s.ts).dt.month
    mom = {}
    for i in inv.investment_id:
        u = usage[usage.investment_id == i].copy()
        u["m"] = u.month.str[-2:].astype(int)
        uu = u.groupby("m").agg(assigned=("licences_assigned", "sum"), active=("active_users", "sum"), sessions=("sessions", "sum"))
        recs = s[(s.investment_id == i) & s.record_ids.notna()].groupby("m").record_ids.nunique()
        mom[i] = [{"m": int(m), "sessions": int(uu.loc[m, "sessions"]) if m in uu.index else 0,
                   "active_share": round(float(uu.loc[m, "active"] / uu.loc[m, "assigned"]), 2) if m in uu.index and uu.loc[m, "assigned"] else 0,
                   "records": int(recs.get(m, 0))} for m in range(1, 13)]
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
    prove = {
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
         "decision": "Consolidate into one build", "avoided": 18000},
        {"id": "inv_chatgpt_ent", "name": "ChatGPT Enterprise (sales seats)", "cost": 54000, "level": "OKR only / unaligned",
         "impact": f"Active seats fell from {round(100*act_first)}% to {round(100*act_last)}%; {100-round(100*chat.aligned.mean())}% of sessions serve no stated goal",
         "method": "Momentum and purpose only; no record trace to compare", "confidence": "Low",
         "decision": "Right-size seats at renewal", "avoided": int(54000 * seat_cut)},
    ]
    out = {"discover": discover, "map": mapping, "drivers": drivers, "trace": trace, "momentum": mom,
           "driver_trend": driver_trend, "prove": prove, "find": find, "report": report,
           "totals": {"ai_spend": int(inv.annual_cost_usd.sum()), "avoided": sum(r["avoided"] for r in report)}}
    DEMO.mkdir(parents=True, exist_ok=True)
    (DEMO / "demo_data.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({"drivers": drivers, "sdr": prove["sdr"]["cycle"], "sdr_range": [prove["sdr"]["lo"], prove["sdr"]["hi"]],
                      "prep": prove["prep"], "dealsum": prove["dealsum"], "find": find, "kr3": kr3_rows,
                      "chat": [act_first, act_last], "totals": out["totals"]}, indent=1, default=str))


if __name__ == "__main__":
    main()
