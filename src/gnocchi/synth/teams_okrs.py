"""Teams and team OKRs.

Teams: each owner is assigned to one team per year by where most of their deals sit
(EMEA, APAC, LATAM, NA East, NA West). Gnocchi works at team level only; owner ids are
used solely to roll deals up to teams.

OKRs: one OKR set per team for 2025, built from the real per-owner annual targets
(summed to team) and the 2024 actuals. Key results point at CRM concepts in words only;
mapping them to fields is the semantic layer's job, so the gold mapping is written
separately for checking.
"""
from __future__ import annotations

import pandas as pd

from .common import SHIFT, OUT, SRC, YEARS, load_crm, team_of_region, write_json


def build_teams() -> pd.DataFrame:
    rows = []
    for y in YEARS:
        crm = load_crm(y)
        crm["team"] = [team_of_region(r, r2) for r, r2 in zip(crm["region"], crm["region (2)"])]
        top = (crm.groupby(["opportunity owner", "team"]).size().reset_index(name="n")
               .sort_values("n", ascending=False).drop_duplicates("opportunity owner"))
        for _, r in top.iterrows():
            rows.append({"year": y, "owner_id": r["opportunity owner"], "team": r["team"]})
    teams = pd.DataFrame(rows).sort_values(["year", "team", "owner_id"])
    OUT.mkdir(parents=True, exist_ok=True)
    teams.to_csv(OUT / "teams.csv", index=False)
    return teams


def build_okrs(teams: pd.DataFrame) -> None:
    targets = pd.read_excel(SRC / "personal_targets.xlsx")
    t24 = targets[targets.year == 2025 - SHIFT].merge(
        teams[teams.year == 2025], left_on="opportunity owner", right_on="owner_id")
    team_target = t24.groupby("team")["annual_target"].sum().round(-4)

    crm23 = load_crm(2024)
    crm23 = crm23.merge(teams[teams.year == 2024], left_on="opportunity owner", right_on="owner_id")
    new23 = crm23[crm23.type == "New"].groupby("team")["amount (converted)"].sum()
    ren23 = crm23[crm23.type == "Renewal"].groupby("team").size()

    okrs, gold = [], []
    for team, target in team_target.items():
        new_goal = round(float(new23.get(team, 0)) * 1.25, -4)
        kr = [
            {"kr_id": f"{team}-KR1", "text": f"Book ${target/1e6:.1f}M in total bookings in 2025",
             "target": float(target), "unit": "USD"},
            {"kr_id": f"{team}-KR2", "text": f"Grow new-logo bookings 25% to ${new_goal/1e6:.2f}M",
             "target": float(new_goal), "unit": "USD"},
            {"kr_id": f"{team}-KR3", "text": "Cut the new-business sales cycle to under 60 days, median",
             "target": 60, "unit": "days"},
            {"kr_id": f"{team}-KR4", "text": f"Renew at least {int(ren23.get(team, 0))} contracts",
             "target": int(ren23.get(team, 0)), "unit": "deals"},
            {"kr_id": f"{team}-KR5", "text": "Raise customer satisfaction with onboarding",
             "target": None, "unit": "score"},
        ]
        okrs.append({"team": team, "year": 2025,
                     "objective": "Grow the business with faster new-logo wins and strong renewals",
                     "key_results": kr})
        gold += [
            {"kr_id": kr[0]["kr_id"], "record": "Opportunity", "fields": "sum(amount (converted)) where sales stage = Closed Won", "time_slice": "year"},
            {"kr_id": kr[1]["kr_id"], "record": "Opportunity", "fields": "sum(amount (converted)) where type = New and Closed Won", "time_slice": "quarter"},
            {"kr_id": kr[2]["kr_id"], "record": "Opportunity + stage history", "fields": "median(close date - created date) where type = New", "time_slice": "month"},
            {"kr_id": kr[3]["kr_id"], "record": "Opportunity", "fields": "count where type = Renewal and Closed Won", "time_slice": "quarter"},
            {"kr_id": kr[4]["kr_id"], "record": None, "fields": None, "time_slice": None, "note": "blind spot: no CRM record holds onboarding satisfaction"},
        ]
    write_json(OUT / "okrs_2025.json", okrs)
    write_json(OUT / "_gold_okr_crm_mapping.json", gold)


def main() -> None:
    teams = build_teams()
    build_okrs(teams)
    print("teams:", teams.groupby(["year", "team"]).size().to_dict())


if __name__ == "__main__":
    main()
