"""Context Engineer agent: builds the context window under a token budget.

Pipeline (no LLM calls, deterministic, millisecond cost):
  1. de-duplicate paragraphs (same title / near-identical text)
  2. split into sentences and score every sentence:
        score = w_c * retrieval score of its paragraph (normalised)
              + w_o * idf-weighted overlap with the query/sub-queries
              + w_b * bridge bonus  (sentence mentions the title of another retrieved paragraph)
              + w_f * first-sentence bonus (defines the entity)
              + w_t * paragraph title appears in the question
  3. greedily keep highest-scoring sentences until the budget is spent
  4. restore original sentence order inside each paragraph and order paragraphs so the
     strongest evidence sits at the start and the end of the window ("lost in the middle").
"""
from __future__ import annotations
import math, re
from collections import defaultdict
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from .schemas import Chunk, EngineeredContext

W = dict(chunk=0.5, overlap=1.0, bridge=0.7, first=0.3, title_q=0.4)
_tok = re.compile(r"[a-z0-9]+")


def est_tokens(text: str) -> int:
    return math.ceil(len(text.split()) * 1.3)


def tokenize(text: str) -> list[str]:
    return [t for t in _tok.findall(text.lower()) if t not in ENGLISH_STOP_WORDS]


def norm_title(t: str) -> str:
    return re.sub(r"\s*\(.*?\)", "", t).strip().lower()


def _dedupe(chunks: list[Chunk]) -> list[Chunk]:
    best: dict[str, Chunk] = {}
    for c in chunks:
        if c.title not in best or c.score > best[c.title].score:
            best[c.title] = c
    out, seen = [], []
    for c in sorted(best.values(), key=lambda x: -x.score):
        toks = set(tokenize(" ".join(c.sentences)))
        if any(len(toks & s) / max(1, len(toks | s)) > 0.8 for s in seen):
            continue
        seen.append(toks)
        out.append(c)
    return out


def _render(title: str, sents: list[str]) -> str:
    return f"[{title}] " + " ".join(sents)


def engineer_context(chunks: list[Chunk], queries: list[str], budget: int, retriever=None) -> EngineeredContext:
    if budget <= 0 or not chunks:
        return EngineeredContext(n_input_chunks=len(chunks))
    kept = _dedupe(chunks)
    max_s = max((c.score for c in kept), default=0.0) or 1.0
    qtext = " ".join(queries).lower()
    q_terms = set(tokenize(qtext))
    idf = retriever.idf if retriever is not None else (lambda t: 1.0)
    q_mass = sum(idf(t) for t in q_terms) + 1e-9
    nts = {c.title: norm_title(c.title) for c in kept}

    scored = []
    total_sents = 0
    for ci, c in enumerate(kept):
        cn = max(c.score, 0.0) / max_s
        t_in_q = 1.0 if nts[c.title] and nts[c.title] in qtext else 0.0
        for si, s in enumerate(c.sentences):
            total_sents += 1
            st = set(tokenize(s))
            ov = sum(idf(t) for t in st & q_terms) / q_mass
            sl = s.lower()
            bridge = 1.0 if any(nt and nt in sl for t2, nt in nts.items() if t2 != c.title) else 0.0
            sc = (W["chunk"] * cn + W["overlap"] * ov + W["bridge"] * bridge
                  + W["first"] * (1.0 if si == 0 else 0.0) + W["title_q"] * t_in_q)
            scored.append((sc, ci, si))
    scored.sort(key=lambda x: -x[0])

    sel: dict[int, set] = defaultdict(set)
    best: dict[int, float] = {}
    used = 0
    for sc, ci, si in scored:
        c = kept[ci]
        cost = est_tokens(c.sentences[si]) + (0 if ci in sel else est_tokens(c.title) + 2)
        if used + cost > budget:
            continue
        sel[ci].add(si)
        best[ci] = max(best.get(ci, 0.0), sc)
        used += cost

    if not sel:  # budget smaller than any sentence: keep a truncated best sentence
        _, ci, si = scored[0]
        words = kept[ci].sentences[si].split()[: max(5, int(budget / 1.3) - 3)]
        text = _render(kept[ci].title, [" ".join(words)])
        return EngineeredContext(text=text, titles=[kept[ci].title], tokens=est_tokens(text),
                                 n_input_chunks=len(kept), n_dropped_sentences=total_sents - 1)

    order = sorted(sel, key=lambda ci: -best[ci])
    arranged = order[0::2] + order[1::2][::-1]  # strongest first, second strongest last
    parts = [_render(kept[ci].title, [kept[ci].sentences[i] for i in sorted(sel[ci])]) for ci in arranged]
    text = "\n".join(parts)
    n_sel = sum(len(v) for v in sel.values())
    return EngineeredContext(text=text, titles=[kept[ci].title for ci in arranged], tokens=est_tokens(text),
                             n_input_chunks=len(kept), n_dropped_sentences=total_sents - n_sel)


def raw_context(chunks: list[Chunk], k: int, budget: int | None = None) -> EngineeredContext:
    """Baseline context: top-k whole paragraphs in rank order (optionally hard-truncated)."""
    if not chunks or k <= 0:
        return EngineeredContext(n_input_chunks=len(chunks))
    top = sorted({c.title: c for c in sorted(chunks, key=lambda x: x.score)}.values(), key=lambda x: -x.score)[:k]
    parts, titles, used = [], [], 0
    for c in top:
        t = _render(c.title, c.sentences)
        tk = est_tokens(t)
        if budget is not None and used + tk > budget:
            remain = budget - used
            if remain > 8:
                t = " ".join(t.split()[: int(remain / 1.3)])
                parts.append(t)
                titles.append(c.title)
                used += est_tokens(t)
            break
        parts.append(t)
        titles.append(c.title)
        used += tk
    text = "\n".join(parts)
    return EngineeredContext(text=text, titles=titles, tokens=est_tokens(text), n_input_chunks=len(chunks))
