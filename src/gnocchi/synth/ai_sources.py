"""Synthetic AI sources: the investment register, telemetry sessions, admin usage and cost.

AI investments in 2024 (one per tool, team-level):
  * inv_ai_sdr        bought, acting   identity svc_ai_sdr      EMEA from Apr, NA East from Jul
  * inv_meeting_prep  bought, assisting via SSO               all teams from Jul, usage varies
  * inv_chatgpt_ent   bought, standalone via SSO              all teams all year, fading use
  * inv_dealsum_emea  built,  acting   identity svc_dealsum_emea  EMEA from Mar  (overlap)
  * inv_dealsum_nae   built,  acting   identity svc_dealsum_nae   NA East from May (overlap)
  No coding assistant: RevOps builders in EMEA and NA East build both deal-summary bots with
  ChatGPT Enterprise. Their sessions carry no repo id; Gnocchi links them to the repo commits
  (Git/CI source) by team, timing and text -> build lineage, and the same bot built twice.
Telemetry sessions carry the prompt text and the real opportunity ids they touched. The
purpose is not in the telemetry: Gnocchi infers it from the text (similarity to operational
drivers and key results). That inference is a hypothesis; joins to records are by key.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .common import OUT, rng, update_ground_truth, write_json

INVESTMENTS = [
    {"investment_id": "inv_ai_sdr", "name": "AI SDR agent", "kind": "bought", "ai_type": "acting",
     "identity": "svc_ai_sdr", "annual_cost_usd": 96000, "teams": ["EMEA", "NA East"]},
    {"investment_id": "inv_meeting_prep", "name": "Meeting-prep assistant", "kind": "bought", "ai_type": "assisting",
     "identity": "sso_user", "annual_cost_usd": 38000, "teams": ["EMEA", "NA East", "NA West", "LATAM", "APAC"]},
    {"investment_id": "inv_chatgpt_ent", "name": "ChatGPT Enterprise (sales seats)", "kind": "bought", "ai_type": "standalone",
     "identity": "sso_user", "annual_cost_usd": 54000, "teams": ["EMEA", "NA East", "NA West", "LATAM", "APAC"]},
    {"investment_id": "inv_dealsum_emea", "name": "Deal-summary bot (EMEA build)", "kind": "built", "ai_type": "acting",
     "identity": "svc_dealsum_emea", "annual_cost_usd": 18000, "teams": ["EMEA"]},
    {"investment_id": "inv_dealsum_nae", "name": "Deal-summary bot (NA East build)", "kind": "built", "ai_type": "acting",
     "identity": "svc_dealsum_nae", "annual_cost_usd": 22000, "teams": ["NA East"]},
]
# builds: coding sessions -> repo -> CI deploy of the bot identity (the Git/CI source)
BUILDS = [
    {"repo": "revops-emea/deal-notes-summarizer", "team": "EMEA", "builds": "inv_dealsum_emea",
     "deploys_identity": "svc_dealsum_emea", "start": "2024-01-08", "deployed_at": "2024-03-01", "engineer_days": 34},
    {"repo": "salesops-nae/opp-summary-bot", "team": "NA East", "builds": "inv_dealsum_nae",
     "deploys_identity": "svc_dealsum_nae", "start": "2024-03-11", "deployed_at": "2024-05-01", "engineer_days": 29},
]
DAY_RATE = 700  # loaded engineer cost per day (assumption)
COMMIT_MSGS = {
    "EMEA": ["Add trigger: send new opp notes to summarizer", "Summarize opportunity notes into 3-bullet status",
             "Retry and rate limit for notes summary job", "Tests for opportunity notes summarizer",
             "Connected app config: write summary note to opportunity", "Fix duplicate summary notes on the same opportunity"],
    "NA East": ["Apex trigger pushes opportunity notes to summary API", "Python summary of opportunity activity notes (3 bullets)",
                "Retry logic and throttling for opportunity summary job", "Opportunity summary bot tests",
                "Connected app so the bot can post summary notes", "Fix bot writing the same note twice"],
}
PURPOSES_CHATGPT = [("Drafting: internal email", 0.26), ("Research: account background", 0.14),
                    ("Summarize: meeting notes", 0.12), ("Personal productivity", 0.16),
                    ("Drafting: proposal text", 0.07), ("Analysis: pipeline and cycle", 0.10),
                    ("Strategy: account and territory plans", 0.08), ("Development: formulas and scripts", 0.07)]
# purpose -> work category (ground truth for the purpose classifier)
PURPOSE_CATEGORY = {"Outreach: first touch": "Writing", "Research: account and contact": "Research",
                    "Research: meeting prep brief": "Research", "Research: account background": "Research",
                    "Summarize: deal notes": "Summarization", "Summarize: meeting notes": "Summarization",
                    "Drafting: internal email": "Writing", "Drafting: proposal text": "Writing",
                    "Personal productivity": "Admin", "Analysis: pipeline and cycle": "Analysis",
                    "Strategy: account and territory plans": "Strategy", "Development: formulas and scripts": "Development",
                    "Development: build deal-summary bot": "Development"}
SEATS = {"EMEA": 12, "NA East": 12, "NA West": 10, "LATAM": 9, "APAC": 4}


def sessions(opps: pd.DataFrame, ev: pd.DataFrame, g: np.random.Generator) -> pd.DataFrame:
    rows, n = [], 0

    def add(ts, inv, identity, team, purpose, records, writes, tokens):
        nonlocal n
        n += 1
        rows.append({"session_id": f"S-{n:06d}", "ts": ts, "investment_id": inv, "identity": identity,
                     "team": team, "purpose": purpose, "record_ids": records, "writes": writes, "tokens": int(tokens)})

    # AI SDR: one session per first-touch it wrote, plus research sessions
    for _, e in ev[(ev.changed_by == "svc_ai_sdr")].iterrows():
        team = opps.loc[e.record_id, "team"]
        add(e.ts, "inv_ai_sdr", "svc_ai_sdr", team, "Outreach: first touch", e.record_id,
            "Activity (Task)", g.normal(1800, 300))
        add(e.ts, "inv_ai_sdr", "svc_ai_sdr", team, "Research: account and contact", e.record_id,
            "none", g.normal(2600, 500))
    # AI SDR also works deals still open at year end (they close in 2025, outside this extract)
    starts = {"EMEA": 4, "NA East": 7}
    for team, start in starts.items():
        for month in range(max(start, 10), 13):
            for k in range(int(g.integers(8, 14))):
                ts = pd.Timestamp(2024, month, int(g.integers(1, 28)), int(g.integers(8, 18)))
                oid = f"open_{team.replace(' ', '').lower()}_{month:02d}_{k:02d}"
                add(ts.strftime("%Y-%m-%dT%H:%M"), "inv_ai_sdr", "svc_ai_sdr", team, "Outreach: first touch",
                    oid, "Activity (Task)", g.normal(1800, 300))
    # Meeting-prep: one session per brief attached
    for _, e in ev[ev.get("created_with", pd.Series(dtype=str)).eq("Meeting-prep assistant")].iterrows():
        team = opps.loc[e.record_id, "team"]
        add(e.ts, "inv_meeting_prep", "sso_user", team, "Research: meeting prep brief", e.record_id,
            "Doc linked to opportunity", g.normal(6200, 900))
    # Deal-summary bots (overlapping builds): summarize notes on open deals in their team
    for inv, ident, team, start in [("inv_dealsum_emea", "svc_dealsum_emea", "EMEA", "2024-03-01"),
                                    ("inv_dealsum_nae", "svc_dealsum_nae", "NA East", "2024-05-01")]:
        pool = opps[(opps.team == team) & (pd.to_datetime(opps.close_date) >= start)]
        for oid, o in pool.sample(frac=0.6, random_state=int(g.integers(1e6))).iterrows():
            ts = pd.Timestamp(o.close_date) - pd.Timedelta(days=int(g.integers(3, 20)))
            add(ts.strftime("%Y-%m-%dT%H:%M"), inv, ident, team, "Summarize: deal notes", oid,
                "Note on opportunity", g.normal(2200, 400))
    # Coding assistant: RevOps builders write the deal-summary bots (a burst before each
    # launch, then maintenance); every session commits to the repo that deploys the bot
    for b in BUILDS:
        start, live = pd.Timestamp(b["start"]), pd.Timestamp(b["deployed_at"])
        days = (live - start).days
        for _ in range(int(g.integers(110, 130))):
            ts = start + pd.Timedelta(days=int(g.integers(0, days)), hours=int(g.integers(8, 19)))
            add(ts.strftime("%Y-%m-%dT%H:%M"), "inv_chatgpt_ent", "sso_user", b["team"], "Development: build deal-summary bot",
                None, "none", g.normal(3600, 700))
            rows[-1]["_repo"] = b["repo"]
        for month in range(live.month, 13):
            for _ in range(int(g.integers(2, 6))):
                ts = pd.Timestamp(2024, month, int(g.integers(1, 28)), int(g.integers(8, 18)))
                add(ts.strftime("%Y-%m-%dT%H:%M"), "inv_chatgpt_ent", "sso_user", b["team"], "Development: build deal-summary bot",
                    None, "none", g.normal(3000, 600))
                rows[-1]["_repo"] = b["repo"]
    # ChatGPT Enterprise: usage fades through the year, mostly not tied to records
    for month in range(1, 13):
        for team, seats in SEATS.items():
            monthly = int(seats * 22 * max(0.25, 1.1 - month * 0.07))
            for _ in range(monthly):
                p = g.choice(len(PURPOSES_CHATGPT), p=[w for _, w in PURPOSES_CHATGPT])
                purpose = PURPOSES_CHATGPT[p][0]
                ts = pd.Timestamp(2024, month, int(g.integers(1, 28)), int(g.integers(8, 18)))
                add(ts.strftime("%Y-%m-%dT%H:%M"), "inv_chatgpt_ent", "sso_user", team, purpose, None,
                    "none", g.normal(900, 250))
    return pd.DataFrame(rows).sort_values("ts")


ACCOUNTS = ["Northgate Transit", "Harbor County", "Ridgeview Health", "Coastal Ports Authority", "Summit Retail",
            "Lakeside University", "Metro Water", "Pinecrest Logistics", "Granite Bank", "Bayline Energy"]
ROLES = ["head of security operations", "IT director", "CIO", "facilities manager", "procurement lead"]
PAINS = ["slow incident response", "too many disconnected cameras", "audit findings on access logs",
         "rising guard costs", "a coming tender"]
TEMPLATES = {
    "Outreach: first touch": [
        "Write a first-touch email to the {role} at {acct} about {pain}. Keep it under 120 words with one clear ask.",
        "Draft a short intro message for a new inbound lead from {acct} who requested a demo yesterday.",
        "Create a 3-step outreach sequence for net-new prospects like {acct}, opening on {pain}.",
    ],
    "Research: account and contact": [
        "Find the decision makers at {acct} and their likely priorities before I reach out.",
        "Who is the {role} at {acct} and what initiatives have they announced this year?",
    ],
    "Research: meeting prep brief": [
        "Prepare a briefing for tomorrow's call with {acct}: attendees, open issues, likely objections and an agenda.",
        "Build a one-page prep brief for my customer meeting with the {role} at {acct}, including what they care about.",
        "Before the meeting with {acct}, summarize their account history and the questions to ask.",
    ],
    "Summarize: deal notes": [
        "Summarize the latest notes on the {acct} opportunity into a status line.",
        "Condense the activity notes for the {acct} deal into three bullets.",
    ],
    "Drafting: internal email": [
        "Rewrite this email to my manager about the holiday coverage schedule so it sounds friendlier.",
        "Draft a short note to the team about the quarterly offsite logistics.",
        "Polish my message to finance about an expense report correction.",
    ],
    "Research: account background": [
        "What does {acct} do, how big are they, and who are their main competitors?",
        "Give me background on {acct}'s industry and recent news before the meeting.",
    ],
    "Summarize: meeting notes": [
        "Summarize these internal team meeting notes into action items.",
        "Turn the notes from our weekly sales sync into a short recap.",
    ],
    "Personal productivity": [
        "Help me plan my week around three priorities.",
        "Convert this table into bullet points.",
        "Fix the grammar in my LinkedIn post about our charity run.",
        "Explain the difference between a pivot table and a lookup.",
    ],
    "Analysis: pipeline and cycle": [
        "Analyze our Q3 pipeline by stage and flag the deals most likely to slip.",
        "Compare win rates and average deal size by vertical for the last two quarters and explain the gap.",
        "Which segments have the longest sales cycle in this export of closed deals, and what correlates with it?",
    ],
    "Strategy: account and territory plans": [
        "Build an account plan to grow {acct} next year, with expansion targets and the stakeholders to win.",
        "Outline a territory strategy for hitting our new-logo bookings target with fewer reps.",
        "Develop a competitive strategy to win the {acct} tender against two incumbents.",
    ],
    "Development: formulas and scripts": [
        "Write a Salesforce report formula that calculates days in each stage.",
        "Write a Python script to dedupe the lead export CSV and normalize phone numbers.",
        "Debug this Excel macro that merges the weekly forecast sheets.",
    ],
    "Development: build deal-summary bot": [
        "Write an Apex trigger that sends new opportunity activity notes to the summarization service.",
        "Write a Python function that condenses an opportunity's activity notes into a three-bullet status.",
        "Add retries and rate limiting to the job that summarizes opportunity notes.",
        "Write unit tests for the opportunity notes summarizer.",
        "Set up the connected app so the bot can write the summary note back to the opportunity.",
        "Debug why the summarizer posts duplicate notes on the same opportunity.",
    ],
    "Development: build deal-summary bot (NA East wording)": [
        "Create an Apex trigger to push new opportunity notes to our summary API.",
        "Python: summarize all activity notes on an opportunity into three short bullets.",
        "Add retry logic and throttling to the opportunity summary job.",
        "Write tests for the opportunity summary bot.",
        "Configure a connected app so the bot can post its summary note to the opportunity record.",
        "The summary bot writes the same note twice on one opportunity. Help me debug it.",
    ],
    "Drafting: proposal text": [
        "Draft the executive summary for a proposal to {acct} covering pricing options and the rollout plan.",
        "Write the pricing and implementation section of our quote for {acct}.",
    ],
}


def add_prompts(sess: pd.DataFrame, g: np.random.Generator) -> pd.DataFrame:
    """Give every session the text a tool's telemetry would hold. The purpose label is
    removed from the telemetry and kept only in the ground truth: Gnocchi must infer it."""
    prompts = []
    tpl = []
    repos = sess["_repo"] if "_repo" in sess else pd.Series([None] * len(sess), index=sess.index)
    for p, rec in zip(sess.purpose, repos):
        if isinstance(rec, str) and rec.startswith("salesops-nae/"):
            p = "Development: build deal-summary bot (NA East wording)"
        k = int(g.integers(len(TEMPLATES[p])))
        tpl.append(k)
        t = TEMPLATES[p][k]
        prompts.append(t.format(acct=ACCOUNTS[int(g.integers(len(ACCOUNTS)))],
                                role=ROLES[int(g.integers(len(ROLES)))],
                                pain=PAINS[int(g.integers(len(PAINS)))]))
    sess = sess.copy()
    sess["prompt"] = prompts
    sess["_tpl"] = tpl
    return sess


def admin_usage(sess: pd.DataFrame) -> pd.DataFrame:
    sess = sess.copy()
    sess["month"] = pd.to_datetime(sess.ts).dt.month
    rows = []
    for inv in INVESTMENTS:
        for team in inv["teams"]:
            for m in range(1, 13):
                s = sess[(sess.investment_id == inv["investment_id"]) & (sess.team == team) & (sess.month == m)]
                seats = SEATS[team] if inv["identity"] == "sso_user" else 1
                active = min(seats, int(np.ceil(len(s) / 20))) if inv["identity"] == "sso_user" else int(len(s) > 0)
                rows.append({"investment_id": inv["investment_id"], "team": team, "month": f"2024-{m:02d}",
                             "licences_assigned": seats, "active_users": active, "sessions": len(s)})
    return pd.DataFrame(rows)


def main() -> None:
    g = rng(2)
    opps = pd.read_csv(OUT / "crm_opportunity_history_2024.csv").set_index("opportunity_id")
    ev = pd.read_csv(OUT / "crm_change_log_2024.csv")
    pd.DataFrame(INVESTMENTS).assign(teams=lambda d: d.teams.str.join(";")).to_csv(OUT / "ai_investments.csv", index=False)
    sess = add_prompts(sessions(opps, ev, g), g)
    sess["category"] = sess.purpose.map(PURPOSE_CATEGORY)
    sess[["session_id", "purpose", "category", "_repo"]].rename(columns={"_repo": "repo"}).to_csv(
        OUT / "_ground_truth_session_purpose.csv", index=False)
    # Git/CI source: commits follow most build sessions within a day or so
    commits = []
    for _, r in sess[sess["_repo"].notna()].iterrows():
        if g.random() < 0.7:
            ts = pd.Timestamp(r.ts) + pd.Timedelta(minutes=int(g.integers(30, 1800)))
            commits.append({"ts": ts.strftime("%Y-%m-%dT%H:%M"), "repo": r["_repo"], "team": r.team,
                            "message": COMMIT_MSGS[r.team][r["_tpl"]]})
    commits = pd.DataFrame(commits).sort_values("ts")
    commits.to_csv(OUT / "code_commits.csv", index=False)
    sess.drop(columns=["purpose", "category", "_repo", "_tpl"]).to_json(OUT / "ai_telemetry_2024.jsonl", orient="records", lines=True)
    admin_usage(sess).to_csv(OUT / "ai_admin_usage_2024.csv", index=False)
    # Git/CI source: one row per build repo
    build_log = pd.DataFrame([{**{k: v for k, v in b.items() if k != "builds"},
                               "commits": int((commits.repo == b["repo"]).sum()),
                               "first_commit": commits[commits.repo == b["repo"]].ts.min()[:10]} for b in BUILDS])
    build_log.to_csv(OUT / "code_build_log.csv", index=False)
    print("sessions", len(sess), sess.investment_id.value_counts().to_dict())
    update_ground_truth("ai_sources", {
        "overlap": ["inv_dealsum_emea", "inv_dealsum_nae"],
        "fading": "inv_chatgpt_ent (usage declines ~7% a month; mostly unaligned purposes)",
        "acting": ["inv_ai_sdr", "inv_dealsum_emea", "inv_dealsum_nae"],
        "assisting": ["inv_meeting_prep"], "standalone": ["inv_chatgpt_ent (incl. the bot builders' coding)"],
        "build_lineage": {b["repo"]: b["builds"] for b in BUILDS}, "duplicate_build": "same templates in both repos",
        "day_rate": DAY_RATE})


if __name__ == "__main__":
    main()
