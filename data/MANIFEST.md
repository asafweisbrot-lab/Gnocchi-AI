# Data manifest

Every source the demo uses, what it holds, and whether it is real or synthetic.
Gnocchi works at team level: owner ids are used only to roll deals up to teams.

## Real (anonymized) — `data/source/`

Copied from the anonymized data Asaf brought into Incentra. No Incentra code was copied.
The reverse keys never left Asaf's machine.

| File | Holds | Used for |
|---|---|---|
| `crm_2022.xlsx` … `crm_2025.xlsx` | Closed-won opportunities (619 / 641 / 695 / 276; 2025 = H1). Owner id, opportunity id, account id, type (New / Expansion / Upsell / Renewal), amount, close date, region, country/state, market, vertical, tech partner | The outcome records. The demo uses 2023 (baseline) and 2024 (AI year) |
| `personal_targets.xlsx` | Annual target per owner and year (2022 real; 2023+ carried forward) | Summed to team targets for the OKRs |
| `strategy_mapping.json` | Comp plan and the CRM field map | Field names for the semantic layer |
| `kpi_definitions.json` | KPI definitions (new business, expansion, renewal, deal size) | KPI wording for the semantic layer |

## Synthetic — `data/synthetic/` (built by `src/gnocchi/synth/`)

| File | Holds | Why it's needed |
|---|---|---|
| `teams.csv` | Owner → team per year (EMEA, APAC, LATAM, NA East, NA West) | Team is Gnocchi's unit |
| `okrs_2024.json` | One objective and five key results per team, built from real targets and 2023 actuals | The OKR lens |
| `_gold_okr_crm_mapping.json` | The correct KR → CRM field mapping, incl. one blind spot | Checks the semantic layer |
| `crm_opportunity_history_{2023,2024}.csv` | Per real deal: created date, cycle days, first-response hours, meetings | The real CRM has only the final state |
| `crm_change_log_{2023,2024}.csv` | Field-history events per deal: stage changes, first touch, meetings, solutions engineer added, materials attached, and who wrote each change (person or agent identity) | History and identity writes |
| `ai_investments.csv` | The 2024 AI register: 5 investments, bought vs built, acting / assisting / standalone, cost, teams | Inventory |
| `ai_telemetry_2024.jsonl` | Sessions: investment, identity, team, purpose, record ids touched, tokens | Purpose and dose |
| `ai_admin_usage_2024.csv` | Monthly licences assigned vs active per tool and team | Momentum (assigned ≠ used) |
| `_ground_truth.json` | The effects planted in the synthetic data | Lets the analysis prove it recovers them |

## Planted ground truth (2024)

- **AI SDR agent** (acting, `svc_ai_sdr`): EMEA from April, NA East from July; other teams never. First response ~6 h → ~1 h; New-deal cycle −8 days.
- **Meeting-prep assistant** (assisting): from July, prep briefs attached at different rates per team (EMEA 70% … APAC 10%); −3 days per attached brief.
- **Solutions engineer before Proposal** (a human action, no AI): −10 days; APAC and LATAM rarely do it, so it should surface as an AI opportunity.
- **Two deal-summary bots** built separately by EMEA and NA East: overlap.
- **ChatGPT Enterprise**: usage fades about 7% a month, mostly on purposes tied to no OKR.
- **Blind spot**: the onboarding-satisfaction key result has no CRM record.

## Known limits

- Closed-won only: win rate can't be studied; drivers are of cycle time and bookings.
- Small teams: NA West (3–4 owners) and APAC (2–4) are thin; results there get wide ranges.
- Everything synthetic is planted, so the demo shows the method works, not that these tools work.

## Session text and purpose
- `ai_telemetry_2024.jsonl` carries a synthetic `prompt` per session; the planted purpose is held out in `_ground_truth_session_purpose.csv` and used only to check the similarity match.
- `code_commits.csv` and `code_build_log.csv` (Git/CI): commits and deploys of the two bot repos. There is no coding assistant: the builders' ChatGPT sessions carry no repo id; Gnocchi links them to commits by team, a 36 h window and text. True links are in `_ground_truth_session_purpose.csv` (`repo`). Planted: NA East rebuilt EMEA's deal-summary bot (paraphrased prompts). Build cost uses an assumed $700 engineer day.
- `_ground_truth_chatgpt_deals.csv`: 2024 deals planted with ChatGPT proposal/account work (−3 days on the cycle). Those sessions name the real (anonymized) account; Gnocchi matches sessions to deals by account, deal type, vertical and the deal's open window. True links are in `_ground_truth_session_purpose.csv` (`opp`).
- Tool work: build sessions before a bot's deploy and growing maintenance after it; plus an unregistered LATAM quote calculator (Apps Script) with no repo or register entry. Maintenance cost assumes 1.5 h per session.
