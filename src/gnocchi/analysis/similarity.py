"""Session purpose -> similarity to operational drivers and key results.

Each telemetry session's text is compared with two sets of targets:
  * operational drivers: the record elements the driver model found to move a target
    (first-touch activity, meeting-prep brief, solutions engineer, proposal, meetings);
  * key results: the team OKR text, widened with the real KPI definition keywords.
The best match above a threshold becomes the session's hypothesis. It says what the
session was aimed at, not what it changed: proof comes from the comparisons.

Encoder: TF-IDF over words and word pairs (runs offline). A sentence-embedding model can
replace it behind the same interface.
"""
from __future__ import annotations

import json

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from gnocchi.synth.common import SRC

DRIVERS = [
    {"id": "D1", "name": "First-touch activity (first response)", "record": "Activity: first-touch task",
     "text": "first touch outreach email intro message new lead inbound requested demo net new prospects reach out outreach sequence ask decision makers who is role priorities initiatives announced initial contact"},
    {"id": "D2", "name": "Meeting-prep brief attached", "record": "Attachment: prep brief",
     "text": "prepare briefing prep brief one page customer meeting tomorrow call attendees agenda objections questions to ask account history background industry competitors recent news before the meeting what they care about"},
    {"id": "D3", "name": "Solutions engineer before Proposal", "record": "Opportunity team: solutions engineer",
     "text": "technical demo solutions engineer architecture requirements technical validation integration"},
    {"id": "D4", "name": "Proposal attached", "record": "Attachment: proposal",
     "text": "proposal quote pricing options executive summary implementation rollout plan"},
    {"id": "D5", "name": "Customer meetings", "record": "Event: customer meeting",
     "text": "schedule customer meeting book a call follow up meeting calendar invite"},
    {"id": "D6", "name": "Deal-notes summary on the opportunity", "record": "Note: deal summary",
     "text": "summarize latest notes opportunity deal status line condense activity notes bullets deal notes summary"},
]


def kr_targets(okrs: list, team: str = "EMEA") -> list:
    kpi = {t["kpi_id"]: t["keywords"] for t in json.loads((SRC / "kpi_definitions.json").read_text())["targets"]}
    extra = {1: kpi.get("deal_size", ""), 2: kpi.get("new_business", ""), 3: "sales cycle faster close days velocity",
             4: kpi.get("renewal", ""), 5: "customer satisfaction onboarding experience"}
    out = []
    for o in okrs:
        if o["team"] != team:
            continue
        for i, kr in enumerate(o["key_results"], start=1):
            out.append({"id": f"KR{i}", "name": kr["text"], "text": kr["text"] + " " + extra.get(i, "")})
    return out


def match(sessions_text: list[str], targets: list[dict], threshold: float = 0.08):
    vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True)
    vec.fit(list(sessions_text) + [t["text"] for t in targets])
    S = vec.transform(sessions_text)
    T = vec.transform([t["text"] for t in targets])
    sim = cosine_similarity(S, T)
    best = sim.argmax(axis=1)
    score = sim[np.arange(len(best)), best]
    ids = [targets[b]["id"] if s >= threshold else None for b, s in zip(best, score)]
    return ids, score, sim
