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

## Run the demo analysis and page

```bash
PYTHONPATH=src python -m gnocchi.analysis.demo   # -> data/demo/demo_data.json
python demo/build_page.py                         # -> demo/gnocchi_demo.html
# optional judge experiment (LiteLLM): set GNOCCHI_JUDGE_MODEL and the provider key, then
# PYTHONPATH=src python -m gnocchi.analysis.judge && rerun the two lines above
```

## Principles

- Team level only. No person is a unit of analysis.
- Session text is matched to drivers and key results to decide what to test; impact is proven by comparison (baselines, rollout waves, usage dose), never by similarity alone.
- New code, written for Gnocchi. Nothing here is copied from Incentra.
