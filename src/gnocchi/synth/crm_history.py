"""Synthetic CRM change history on top of the real closed-won deals.

The real CRM holds only the final state of each won deal (owner, type, amount, close
date, region...). Gnocchi needs the history: when the deal was created, when it moved
stage, which actions and materials preceded the win, and who (a person or an agent
identity) wrote each change. This generator adds that history to every real deal.

Planted effects (also written to _ground_truth.json):
  * AI SDR agent (acting AI, identity svc_ai_sdr) writes the first-touch activity on New
    deals closing 2+ months after rollout: EMEA from 2025-04, NA East from 2025-07 (staggered rollout; other teams never).
    First response drops from ~6 h to ~1 h; each hour of response adds ~1.6 days, so the
    New-deal cycle shortens by ~8 days through that driver.
  * Meeting-prep assistant (assisting AI): from 2025-07 a prep brief is attached to the
    opportunity at a team-specific rate (dose); each attached brief shortens the cycle ~3 days.
  * Solutions engineer added before Proposal (a human action, no AI): ~10 days shorter
    cycle; APAC and LATAM rarely do it -> an AI opportunity, not an AI effect.
  * ChatGPT on live deals (standalone AI, no record trace): in 2025 ~30% of New, Expansion
    and Upsell deals get proposal and account work in ChatGPT; those deals close ~3 days
    faster. The CRM never sees it: only the session text (account name, timing) links it.
Cycle lengths also depend on deal type and size, plus noise and a Q4 push.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .common import OUT, YEARS, load_crm, rng, update_ground_truth

SDR_START = {"EMEA": pd.Timestamp("2025-04-01"), "NA East": pd.Timestamp("2025-07-01")}
PREP_START = pd.Timestamp("2025-07-01")
PREP_RATE = {"EMEA": 0.70, "NA East": 0.55, "NA West": 0.30, "LATAM": 0.20, "APAC": 0.10}
SE_RATE = {"EMEA": 0.55, "NA East": 0.50, "NA West": 0.45, "LATAM": 0.20, "APAC": 0.15}
BASE_CYCLE = {"New": 78, "Expansion": 48, "Upsell": 40, "Renewal": 30, "Partner Showroom": 30}
GPT_RATE = 0.30
EFFECT = {"chatgpt_deal_cycle_days": -3.0, "days_per_response_hour": 1.6, "sdr_cycle_days_via_response": -8.0, "prep_cycle_days": -3.0, "se_cycle_days": -10.0,
          "resp_hours_before": 6.0, "resp_hours_with_sdr": 1.0}
STAGES = ["Discovery", "Demo", "Proposal", "Negotiation", "Closed Won"]
PROBABILITY = {"Discovery": 10, "Demo": 25, "Proposal": 50, "Negotiation": 75, "Closed Won": 100}
CONTACT_ROLES = ["CIO", "IT director", "Head of security operations", "Procurement lead", "Facilities manager", "CFO"]
SITES = ["main campus", "north depot", "harbor site", "downtown HQ", "west warehouse", "airport terminal", "central station"]
NEXT_STEPS = {
    "Discovery": ["Discovery call with the {role}", "Map current camera estate ({n} cameras)", "Confirm budget owner"],
    "Demo": ["Pilot with {n} cameras at the {site}", "Technical demo for the {role}", "Security review with IT"],
    "Proposal": ["Send revised quote for {n} cameras", "Pricing call with the {role}", "Add the {site} to the scope"],
    "Negotiation": ["Legal redlines on the MSA", "Procurement approval from the {role}", "Board approval expected in {month}"],
}
STAGE_SHARE = [0.0, 0.25, 0.55, 0.8, 1.0]


def enrich(year: int, teams: pd.DataFrame, g: np.random.Generator, gh: np.random.Generator | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    gh = gh or np.random.default_rng(11)
    crm = load_crm(year)
    tmap = teams[teams.year == year].set_index("owner_id")["team"]
    crm["team"] = crm["opportunity owner"].map(tmap)
    opps, events = [], []
    for _, d in crm.iterrows():
        team, typ, close = d["team"], d["type"], d["close date"]
        amt = float(d["amount (converted)"] or 0)
        is_new = typ == "New"
        se = bool(g.random() < SE_RATE.get(team, 0.3)) if typ in ("New", "Expansion") else False
        cycle = BASE_CYCLE.get(typ, 40) + 6 * np.log1p(amt / 50_000) + (6 if close.month == 12 else 0)
        cycle += g.normal(0, 7)
        if se:
            cycle += EFFECT["se_cycle_days"]
        # created date first estimate, then decide AI exposure at creation time
        created = close - pd.Timedelta(days=max(10, int(round(cycle))))
        # exposure is set by team and close period (deals closing 2+ months after the
        # team's rollout were worked with the agent), so it never depends on cycle noise
        sdr = is_new and team in SDR_START and close >= SDR_START[team] + pd.Timedelta(days=60)
        prep = (typ in ("New", "Expansion") and close >= PREP_START
                and bool(g.random() < PREP_RATE.get(team, 0.2)))
        if prep:
            cycle += EFFECT["prep_cycle_days"]
        gpt = year == 2025 and typ in ("New", "Expansion", "Upsell") and bool(g.random() < GPT_RATE)
        if gpt:
            cycle += EFFECT["chatgpt_deal_cycle_days"]
        cycle = max(10, int(round(cycle)))
        created = close - pd.Timedelta(days=cycle)
        resp_h = (max(0.2, g.normal(EFFECT["resp_hours_with_sdr"], 0.4)) if sdr
                  else max(0.5, g.normal(EFFECT["resp_hours_before"], 2.0))) if is_new else None
        if is_new:
            # first response is the driver: each hour slower adds ~1.6 days, so the AI SDR's
            # 6 h -> 1 h moves the cycle by about -8 days through the driver
            cycle += EFFECT["days_per_response_hour"] * (resp_h - EFFECT["resp_hours_before"])
        meetings = int(max(1, round(g.normal(4 if is_new else 2, 1.2))))
        oid = d["opportunity id"]
        opps.append({"opportunity_id": oid, "year": year, "team": team, "type": typ,
                     "amount": amt, "created_date": created.date(), "close_date": close.date(),
                     "cycle_days": cycle, "first_response_hours": None if resp_h is None else round(resp_h, 1),
                     "meetings": meetings, "_gpt": gpt})
        # stage events
        for s, share in zip(STAGES, STAGE_SHARE):
            ts = created + pd.Timedelta(days=int(round(cycle * share)))
            events.append({"ts": ts, "object": "Opportunity", "record_id": oid, "field": "StageName",
                           "old_value": None if s == "Discovery" else STAGES[STAGES.index(s) - 1],
                           "new_value": s, "changed_by": "user"})
        if is_new:
            ts = created + pd.Timedelta(hours=resp_h)
            events.append({"ts": ts, "object": "Activity", "record_id": oid, "field": "Task",
                           "old_value": None, "new_value": "First-touch email",
                           "changed_by": "svc_ai_sdr" if sdr else "user"})
        for k in range(meetings):
            ts = created + pd.Timedelta(days=int(cycle * (0.1 + 0.6 * (k + 1) / (meetings + 1))))
            events.append({"ts": ts, "object": "Activity", "record_id": oid, "field": "Event",
                           "old_value": None, "new_value": "Customer meeting", "changed_by": "user"})
        if se:
            ts = created + pd.Timedelta(days=int(cycle * 0.4))
            events.append({"ts": ts, "object": "OpportunityTeamMember", "record_id": oid, "field": "Role",
                           "old_value": None, "new_value": "Solutions Engineer", "changed_by": "user"})
        if prep:
            ts = created + pd.Timedelta(days=int(cycle * 0.3))
            events.append({"ts": ts, "object": "ContentDocumentLink", "record_id": oid, "field": "Attachment",
                           "old_value": None, "new_value": "Meeting prep brief",
                           "changed_by": "user", "created_with": "Meeting-prep assistant"})
        if typ in ("New", "Expansion", "Upsell"):
            ts = created + pd.Timedelta(days=int(cycle * 0.55))
            events.append({"ts": ts, "object": "ContentDocumentLink", "record_id": oid, "field": "Attachment",
                           "old_value": None, "new_value": "Proposal", "changed_by": "user"})
        # Richer history, drawn from its own generator so the planted effects above don't move:
        # probability with each stage, amount revisions, close-date pushes, next steps, contacts
        events += history_detail(oid, created, cycle, close, amt, gh)
    ev = pd.DataFrame(events).sort_values("ts")
    ev["ts"] = pd.to_datetime(ev["ts"]).dt.strftime("%Y-%m-%dT%H:%M")
    return pd.DataFrame(opps), ev


def history_detail(oid, created, cycle, close, amt, h) -> list[dict]:
    out = []
    def add(ts, obj, field, old, new):
        out.append({"ts": ts, "object": obj, "record_id": oid, "field": field, "old_value": old, "new_value": new, "changed_by": "user"})
    prev = None
    for st, share in zip(STAGES, STAGE_SHARE):
        add(created + pd.Timedelta(days=int(round(cycle * share)), minutes=5), "Opportunity", "Probability", prev, PROBABILITY[st])
        prev = PROBABILITY[st]
    # amount: a first estimate, a revision at proposal, the final figure at negotiation
    a0 = round(max(1000.0, amt * h.uniform(0.6, 1.4)), -3)
    a1 = round(max(1000.0, amt * h.uniform(0.85, 1.15)), -3)
    add(created + pd.Timedelta(hours=2), "Opportunity", "Amount", None, a0)
    add(created + pd.Timedelta(days=int(cycle * 0.55), hours=3), "Opportunity", "Amount", a0, a1)
    add(created + pd.Timedelta(days=int(cycle * 0.8), hours=3), "Opportunity", "Amount", a1, round(amt, 0))
    # close date: an early guess, pushed once or twice toward the real close
    first = created + pd.Timedelta(days=int(cycle * h.uniform(0.55, 0.85)))
    add(created + pd.Timedelta(hours=2), "Opportunity", "CloseDate", None, str(first.date()))
    pushes = int(h.integers(0, 3))
    cur = first
    for k in range(pushes):
        nxt = cur + (close - cur) * ((k + 1) / pushes) if pushes else close
        nxt = pd.Timestamp(nxt).normalize()
        when = min(cur, close) - pd.Timedelta(days=int(h.integers(1, 6)))
        add(max(created, when), "Opportunity", "CloseDate", str(cur.date()), str(nxt.date()))
        cur = nxt
    # next steps, one or two per open stage, with concrete details
    for st, share in zip(STAGES[:-1], STAGE_SHARE[:-1]):
        for _ in range(int(h.integers(1, 3))):
            t = NEXT_STEPS[st][int(h.integers(len(NEXT_STEPS[st])))]
            text = t.format(role=CONTACT_ROLES[int(h.integers(len(CONTACT_ROLES)))], n=int(h.choice([8, 12, 20, 24, 40, 60, 120])),
                            site=SITES[int(h.integers(len(SITES)))], month=(close + pd.Timedelta(days=10)).strftime("%B"))
            add(created + pd.Timedelta(days=int(cycle * (share + h.uniform(0.02, 0.2)))), "Opportunity", "NextStep", None, text)
    # contacts on the deal
    for role in h.choice(CONTACT_ROLES, size=int(h.integers(1, 3)), replace=False):
        add(created + pd.Timedelta(days=int(cycle * h.uniform(0.05, 0.4))), "OpportunityContactRole", "Role", None, str(role))
    return out


def main() -> None:
    teams = pd.read_csv(OUT / "teams.csv")
    g = rng(1)
    gh = rng(11)
    for y in YEARS:
        opps, ev = enrich(y, teams, g, gh)
        if y == 2025:
            opps[opps._gpt][["opportunity_id"]].to_csv(OUT / "_ground_truth_chatgpt_deals.csv", index=False)
        opps.drop(columns=["_gpt"]).to_csv(OUT / f"crm_opportunity_history_{y}.csv", index=False)
        ev.to_csv(OUT / f"crm_change_log_{y}.csv", index=False)
        print(y, "opps", len(opps), "events", len(ev))
    update_ground_truth("crm_history", {"sdr_start": {k: str(v.date()) for k, v in SDR_START.items()},
                                        "prep_start": str(PREP_START.date()), "prep_rate": PREP_RATE,
                                        "se_rate": SE_RATE, "gpt_rate": GPT_RATE, "effects": EFFECT})


if __name__ == "__main__":
    main()
