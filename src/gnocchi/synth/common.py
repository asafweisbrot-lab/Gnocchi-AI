"""Shared helpers for the synthetic source generators.

Real inputs live in data/source/ (anonymized CRM, targets, comp plan, KPI definitions).
Synthetic outputs go to data/synthetic/. Every generator is seeded, so outputs are
reproducible. Planted effects are written to data/synthetic/_ground_truth.json so the
analysis can later be checked against what was put in.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "data" / "source"
OUT = ROOT / "data" / "synthetic"
YEARS = (2024, 2025)          # 2024 = baseline (no AI), 2025 = AI year
# The real CRM holds full years 2022-2024 (2025 only to June). The demo uses the full real
# years 2023 and 2024 and moves every date one year later, so they read as 2024 and 2025.
SHIFT = 1
SEED = 20261102

WEST = {"CA", "WA", "OR", "AZ", "UT", "CO", "NV", "NM", "HI", "ID", "MT", "WY",
        "British Columbia", "Alberta", "Yukon", "Saskatchewan", "Manitoba"}


def rng(offset: int = 0) -> np.random.Generator:
    return np.random.default_rng(SEED + offset)


def load_crm(year: int) -> pd.DataFrame:
    """Real deals for a demo year: the source year is `year - SHIFT`, dates moved forward."""
    df = pd.read_excel(SRC / f"crm_{year - SHIFT}.xlsx")
    df["close date"] = pd.to_datetime(df["close date"]) + pd.DateOffset(years=SHIFT)
    if "year" in df:
        df["year"] = df["year"] + SHIFT
    return df


def team_of_region(region: str, region2: str) -> str:
    if region == "North America":
        return "NA West" if region2 in WEST else "NA East"
    return {"EMEA": "EMEA", "APAC": "APAC", "LATAM": "LATAM"}.get(region, region)


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))


def update_ground_truth(key: str, value) -> None:
    path = OUT / "_ground_truth.json"
    data = json.loads(path.read_text()) if path.exists() else {}
    data[key] = value
    write_json(path, data)
