"""Experiment: a small model as judge, called through LiteLLM.

Stage 1 (in demo.py, word matching) shortlists candidates for each session in a labelled
sample: the top drivers / key results, and the open deals of the session's team. Stage 2
(here) asks a model to pick one of each, or none, with a one-line reason. demo.py then
scores both stages against the planted ground truth on the same sample.

    PYTHONPATH=src python -m gnocchi.analysis.demo      # writes data/demo/judge_sample.json
    GNOCCHI_JUDGE_MODEL=anthropic/claude-haiku-4-5 ANTHROPIC_API_KEY=... \\
        PYTHONPATH=src python -m gnocchi.analysis.judge  # writes data/demo/judge_predictions.json
    PYTHONPATH=src python -m gnocchi.analysis.demo      # merges the results into the report

The model is a LiteLLM string, so the same code runs a hosted model or a local open one
(e.g. ollama/qwen2.5:7b) without changes. With no model set, nothing is called.
"""
from __future__ import annotations

import json
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from gnocchi.synth.common import ROOT

DEMO = ROOT / "data" / "demo"
SYSTEM = (
    "You read one session from a company's AI tools and decide what it serves. "
    "Pick the single best target from the list, or NONE if the session serves none of them. "
    "If deal candidates are listed, pick the deal the session is about, or NONE: a deal matches when the session "
    "names its account; if several deals share that account, pick the one whose deal type matches the session. "
    "Prefer NONE over a weak guess. Answer with JSON only: "
    '{"target": "<id or NONE>", "deal": "<id or NONE>", "reason": "<under 20 words>"}'
)


def prompt(item: dict) -> str:
    lines = [f"Session (team {item['team']}, {item['ts'][:10]}):", item["prompt"], "", "Targets:"]
    lines += [f"- {t['id']}: {t['name']}" for t in item["targets"]] + ["- NONE: serves none of these"]
    if item["deals"]:
        lines += ["", "Deal candidates (open when the session ran):"]
        lines += [f"- {d['id']}: {d['text']}" for d in item["deals"]] + ["- NONE: about none of these deals"]
    return "\n".join(lines)


def judge_one(model: str, item: dict) -> dict:
    import litellm
    t0 = time.time()
    for attempt in range(3):
        try:
            extra = {"api_base": os.environ["GNOCCHI_JUDGE_API_BASE"], "api_key": "local"} if os.environ.get("GNOCCHI_JUDGE_API_BASE") else {}
            r = litellm.completion(model=model, temperature=0, max_tokens=120, **extra,
                                   messages=[{"role": "system", "content": SYSTEM},
                                             {"role": "user", "content": prompt(item)}])
            text = r.choices[0].message.content.strip()
            text = text[text.find("{"): text.rfind("}") + 1]
            out = json.loads(text)
            try:
                cost = float(litellm.completion_cost(completion_response=r))
            except Exception:
                cost = None
            return {"session_id": item["session_id"], "target": None if out.get("target") in (None, "NONE") else out["target"],
                    "deal": None if out.get("deal") in (None, "NONE") else out["deal"], "reason": out.get("reason", ""),
                    "latency_s": round(time.time() - t0, 2), "cost_usd": cost,
                    "tokens": getattr(r, "usage", None) and r.usage.total_tokens}
        except Exception as e:  # noqa: BLE001
            err = str(e)
            time.sleep(2 * (attempt + 1))
    return {"session_id": item["session_id"], "error": err[:200]}


def main() -> None:
    model = os.environ.get("GNOCCHI_JUDGE_MODEL")
    if not model:
        sys.exit("Set GNOCCHI_JUDGE_MODEL (a LiteLLM model string) and the provider's API key.")
    sample = json.loads((DEMO / "judge_sample.json").read_text())
    limit = int(os.environ.get("GNOCCHI_JUDGE_LIMIT", "0"))
    if limit and limit < len(sample["items"]):
        import random
        sample["items"] = random.Random(7).sample(sample["items"], limit)
    with ThreadPoolExecutor(max_workers=int(os.environ.get("GNOCCHI_JUDGE_WORKERS", "6"))) as ex:
        preds = list(ex.map(lambda it: judge_one(model, it), sample["items"]))
    ok = [p for p in preds if "error" not in p]
    summary = {"model": model, "n": len(preds), "errors": len(preds) - len(ok),
               "median_latency_s": round(statistics.median([p["latency_s"] for p in ok]), 2) if ok else None,
               "cost_usd": round(sum(p["cost_usd"] or 0 for p in ok), 4) if ok else None,
               "ran_at": time.strftime("%Y-%m-%d %H:%M")}
    (DEMO / "judge_predictions.json").write_text(json.dumps({"summary": summary, "predictions": preds}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
