# Gnocchi AI

Shows the business impact of every organizational AI investment, team by team, from the
operational records where the business already keeps score.

This repo starts with a demo: one year of real (anonymized) closed-won deals, plus
synthetic CRM history and AI logs, run step by step from discovery to an investment report.

## Layout

- `data/source/` — real, anonymized inputs (see `data/MANIFEST.md`)
- `data/synthetic/` — generated sources, rebuilt by the generators below
- `src/gnocchi/synth/` — synthetic source generators (seeded, reproducible)

## Rebuild the synthetic sources

```bash
pip install -r requirements.txt
PYTHONPATH=src python -m gnocchi.synth.run_all
```

## Principles

- Team level only. No person is a unit of analysis.
- Impact is proven by comparison (baselines, rollout waves, usage dose), never by text similarity.
- New code, written for Gnocchi. Nothing here is copied from Incentra.
