"""Support and HR previews for the demo report: everything synthetic.

For each department: a small synthetic operational database (tickets / requisitions), team
OKRs mapped to fields, an AI inventory with telemetry counts, the top-down / bottom-up link
counts, and one impact result from a staggered rollout. Effects are planted, so the estimate
can be checked against the truth, the same way the Sales tab is.

    PYTHONPATH=src python -m gnocchi.analysis.depts     # writes data/demo/depts.json
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from gnocchi.analysis.demo import boot, ols
from gnocchi.synth.common import ROOT

OUT = ROOT / "data" / "demo" / "depts.json"


def staggered(rng, teams, n, base, sd, start_months, effect, drivers):
    """Records over 2025; a tool reaches each team in a different month (None = never)."""
    rows = []
    for _ in range(n):
        team = rng.choice(teams)
        m = int(rng.integers(1, 13))
        live = start_months[team] is not None and m >= start_months[team]
        x = {k: float(f(rng)) for k, (f, _) in drivers.items()}
        y = base * np.exp(rng.normal(0, sd)) + sum(x[k] * w for k, (_, w) in drivers.items())
        y += effect if live else 0.0
        rows.append({"team": team, "month": m, "ai": int(live), **x, "y": max(0.5, y)})
    return pd.DataFrame(rows)


def estimate(df, xs):
    c = ols(df, "y", ["ai"] + xs, ["team", "month"])
    lo, hi = boot(df, lambda d: ols(d, "y", ["ai"] + xs, ["team", "month"])["ai"][0], n=200)
    drv = {k: round(c[k][0], 2) for k in xs}
    return round(c["ai"][0], 1), round(lo, 1), round(hi, 1), drv


def support(rng):
    teams = ["Tier 1 EMEA", "Tier 1 NA", "Tier 2", "Billing"]
    starts = {"Tier 1 EMEA": 3, "Tier 1 NA": 6, "Tier 2": 9, "Billing": None}   # auto-triage waves
    drivers = {"first_assign_h": (lambda r: r.gamma(2, 2), 0.9), "kb_linked": (lambda r: r.random() < 0.4, -3.0),
               "reopened": (lambda r: r.random() < 0.12, 9.0)}
    df = staggered(rng, teams, 4200, 22, 0.45, starts, -4.0, drivers)
    est, lo, hi, drv = estimate(df, list(drivers))
    scope = len(df)
    matter = int((df.kb_linked.astype(bool) | (df.first_assign_h < 2)).sum())
    traced = int(df.ai.sum())
    return {
        "name": "Support", "crumb": "Reports / AI impact / Support org · FY2025",
        "lead": "Gnocchi read the ticketing system, the support team OKRs and every AI source support runs, and traced each AI tool to the ticket fields that move resolution time.",
        "stats": [["4", "AI investments"], [f"{scope:,}", "tickets in 2025"], ["4", "support teams"], [f"{est:+.1f} h", "resolution time, auto-triage"]],
        "inventory": [
            {"name": "Auto-triage agent", "kind": "Bought", "type": "Direct · agent", "writes": "Queue, priority, first assignee", "cost": 96000, "teams": "Tier 1 EMEA, Tier 1 NA, Tier 2"},
            {"name": "Reply copilot", "kind": "Bought", "type": "Indirect · assists agents", "writes": "Draft reply marked AI-drafted", "cost": 64000, "teams": "All"},
            {"name": "KB article summarizer", "kind": "Built", "type": "Indirect · material", "writes": "Summary attached to KB link", "cost": 18000, "teams": "Tier 2"},
            {"name": "ChatGPT Enterprise", "kind": "Bought", "type": "Indirect · sessions", "writes": "Nothing", "cost": 41000, "teams": "All"}],
        "map": [
            {"kr": "Cut median time to resolution under 24 hours", "fields": ["Resolved at", "Created at"], "duration": True, "action": "Median", "why": "a duration per ticket, then the typical ticket", "rows": ["Status = Resolved"], "date": "Resolved at", "grain": "month", "record": "Ticket"},
            {"kr": "Raise CSAT to 4.5", "fields": ["CSAT score"], "action": "Average", "why": "the mean survey score", "rows": ["Survey answered"], "date": "Resolved at", "grain": "month", "record": "Ticket + survey"},
            {"kr": "Deflect 30% of contacts to self-service", "fields": ["Channel", "Article viewed"], "action": "Share", "why": "deflected ÷ all contacts", "rows": ["Contact created"], "date": "Created at", "grain": "quarter", "record": "Contact + KB views"},
            {"kr": "Keep agent attrition under 15%", "blind": "Attrition lives in the HR system, not in ticketing; connect the HRIS to see it"}],
        "drivers": [{"driver": "First assignment time", "per": "per hour slower", "v": drv["first_assign_h"], "unit": "h"},
                    {"driver": "KB article linked", "per": "when linked", "v": drv["kb_linked"], "unit": "h"},
                    {"driver": "Ticket reopened", "per": "when reopened", "v": drv["reopened"], "unit": "h"}],
        "links": {"scope": scope, "matter": matter, "traced": traced,                   "td_label": "tickets where a field that moves resolution was set", "trace_label": "carry an agent write from auto-triage",
                  "bu_label": "ChatGPT and copilot sessions matched to a ticket by customer, product and timing"},
        "impact": {"tool": "Auto-triage agent", "kr": "Median time to resolution", "est": est, "lo": lo, "hi": hi, "unit": "hours",
                   "truth": -4.0, "method": "Rolled out to Tier 1 EMEA in March, Tier 1 NA in June, Tier 2 in September, never to Billing; each team compared with itself and with the teams not yet live, controlling for team and month.",
                   "n": scope},
        "decisions": [["Auto-triage agent", "Expand to Billing", "Cuts resolution time where it runs; Billing is the one team without it."],
                      ["Reply copilot", "Keep, test CSAT", "Used daily; its effect on CSAT isn't separable yet. Run a 6-week holdout."],
                      ["KB article summarizer", "Right-size", "One team, few sessions; fold into the KB tool at renewal."],
                      ["ChatGPT Enterprise", "Keep", "Shared across departments; support sessions link to tickets, so its share is counted."]],
    }


def hr(rng):
    teams = ["Engineering hiring", "Sales hiring", "G&A hiring", "Campus"]
    starts = {"Engineering hiring": 2, "Sales hiring": 5, "G&A hiring": 8, "Campus": None}   # scheduling agent waves
    drivers = {"first_screen_d": (lambda r: r.gamma(2, 2.5), 0.8), "panel_in_3d": (lambda r: r.random() < 0.45, -4.0),
               "late_feedback": (lambda r: r.random() < 0.3, 5.0)}
    df = staggered(rng, teams, 1100, 38, 0.35, starts, -5.0, drivers)
    est, lo, hi, drv = estimate(df, list(drivers))
    scope = len(df)
    matter = int((df.panel_in_3d.astype(bool) | (df.first_screen_d < 3)).sum())
    traced = int(df.ai.sum())
    return {
        "name": "HR", "crumb": "Reports / AI impact / People team · FY2025",
        "lead": "Gnocchi read the applicant tracking system, the HR team OKRs and every AI source recruiting runs, and traced each AI tool to the requisition fields that move time to hire.",
        "stats": [["4", "AI investments"], [f"{scope:,}", "requisitions in 2025"], ["4", "hiring teams"], [f"{est:+.1f} d", "time to hire, scheduling agent"]],
        "inventory": [
            {"name": "Interview scheduling agent", "kind": "Bought", "type": "Direct · agent", "writes": "Interview events on the candidate", "cost": 54000, "teams": "Engineering, Sales, G&A hiring"},
            {"name": "Resume screening agent", "kind": "Bought", "type": "Direct · agent", "writes": "Candidate stage: screened in / out", "cost": 72000, "teams": "All"},
            {"name": "Job description writer", "kind": "Built", "type": "Indirect · material", "writes": "JD draft attached to the requisition", "cost": 12000, "teams": "G&A hiring"},
            {"name": "ChatGPT Enterprise", "kind": "Bought", "type": "Indirect · sessions", "writes": "Nothing", "cost": 23000, "teams": "All"}],
        "map": [
            {"kr": "Cut median time to hire under 35 days", "fields": ["Offer accepted at", "Requisition opened at"], "duration": True, "action": "Median", "why": "a duration per hire, then the typical hire", "rows": ["Status = Filled"], "date": "Offer accepted at", "grain": "month", "record": "Requisition + candidate"},
            {"kr": "Raise offer acceptance to 85%", "fields": ["Offer status"], "action": "Share", "why": "accepted ÷ offers made", "rows": ["Offer made"], "date": "Offer made at", "grain": "quarter", "record": "Offer"},
            {"kr": "Keep first-year attrition under 10%", "fields": ["Termination date", "Hire date"], "action": "Share", "why": "leavers within a year ÷ hires", "rows": ["Hired in the last 12 months"], "date": "Hire date", "grain": "quarter", "record": "Employee (HRIS)"},
            {"kr": "Improve the interview experience", "blind": "No system records candidate experience; add a post-interview survey to see it"}],
        "drivers": [{"driver": "Days to first screen", "per": "per day slower", "v": drv["first_screen_d"], "unit": "d"},
                    {"driver": "Interview panel within 3 days", "per": "when it happens", "v": drv["panel_in_3d"], "unit": "d"},
                    {"driver": "Late hiring-manager feedback", "per": "when late", "v": drv["late_feedback"], "unit": "d"}],
        "links": {"scope": scope, "matter": matter, "traced": traced,                   "td_label": "requisitions where a field that moves time to hire was set", "trace_label": "carry interview events written by the scheduling agent",
                  "bu_label": "ChatGPT sessions matched to a requisition by role, team and timing"},
        "impact": {"tool": "Interview scheduling agent", "kr": "Median time to hire", "est": est, "lo": lo, "hi": hi, "unit": "days",
                   "truth": -5.0, "method": "Rolled out to Engineering hiring in February, Sales in May, G&A in August, never to Campus; each team compared with itself and with the teams not yet live, controlling for team and month.",
                   "n": scope},
        "decisions": [["Interview scheduling agent", "Expand to Campus", "Shortens time to hire where it runs."],
                      ["Resume screening agent", "Investigate", "Writes stages on every candidate, but screened-in candidates aren't hired faster; check what it optimizes."],
                      ["Job description writer", "Retire", "Used by one team a few times a month; ChatGPT covers it."],
                      ["ChatGPT Enterprise", "Keep", "Shared; recruiting sessions link to requisitions."]],
    }


def main() -> None:
    rng = np.random.default_rng(11)
    out = {"support": support(rng), "hr": hr(rng)}
    OUT.write_text(json.dumps(out, indent=1))
    for k, v in out.items():
        print(k, v["impact"]["est"], v["impact"]["lo"], v["impact"]["hi"], "truth", v["impact"]["truth"], v["drivers"])


if __name__ == "__main__":
    main()
