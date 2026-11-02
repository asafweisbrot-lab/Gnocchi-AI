"""Session text -> work category (research, writing, analysis, strategy, development,
summarization, admin). The first read of the telemetry: what the AI is used for, before
asking which driver or key result it is aimed at.

Encoder: TF-IDF over words and word pairs, nearest category descriptor. A sentence-embedding
model can replace it behind the same interface.
"""
from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

CATEGORIES = [
    {"id": "Research", "text": "find who is what does background research decision makers priorities initiatives announced "
                               "prepare briefing prep brief before the meeting account history industry recent news competitors questions to ask"},
    {"id": "Writing", "text": "write draft rewrite polish email message intro first-touch outreach sequence note proposal "
                              "executive summary section quote friendlier under words ask"},
    {"id": "Analysis", "text": "analyze compare win rates pipeline by stage segments correlates export data trend explain the gap "
                               "flag deals likely to slip average deal size longest cycle"},
    {"id": "Strategy", "text": "strategy plan account plan territory grow expansion targets stakeholders win tender incumbents "
                               "competitive outline bookings target"},
    {"id": "Development", "text": "write formula script python excel macro debug code dedupe normalize csv report formula automate"},
    {"id": "Summarization", "text": "summarize condense notes into bullets status line recap action items activity notes turn the notes"},
    {"id": "Admin", "text": "plan my week priorities convert table bullet points fix grammar linkedin post explain the difference"},
]


def classify(texts: list[str]):
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
    vec.fit(list(texts) + [c["text"] for c in CATEGORIES])
    sim = cosine_similarity(vec.transform(texts), vec.transform([c["text"] for c in CATEGORIES]))
    best = sim.argmax(1)
    return [CATEGORIES[b]["id"] for b in best], sim.max(1)
