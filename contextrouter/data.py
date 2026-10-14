"""Dataset loading. Files are in the original HotpotQA JSON format:
{"_id","question","answer","type","level","supporting_facts":[[title,idx]],"context":[[title,[sent,...]]]}"""
from __future__ import annotations
import json
from pathlib import Path


def load_questions(path: str, n: int | None = None) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        qs = json.load(f)
    return qs[:n] if n else qs


def build_corpus(questions: list[dict]) -> list[dict]:
    """Shared paragraph corpus = union of all paragraphs of the selected questions (gold + distractors)."""
    by_title: dict[str, list[str]] = {}
    for q in questions:
        for title, sents in q["context"]:
            if title not in by_title or len(" ".join(sents)) > len(" ".join(by_title[title])):
                by_title[title] = sents
    return [{"title": t, "sentences": s} for t, s in by_title.items()]


def gold_titles(q: dict) -> list[str]:
    return sorted({t for t, _ in q["supporting_facts"]})
