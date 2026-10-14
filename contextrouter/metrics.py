import re, string
from collections import Counter
import numpy as np


def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(pred: str, gold: str) -> float:
    return float(normalize_answer(pred) == normalize_answer(gold))


def f1_score(pred: str, gold: str) -> float:
    p, g = normalize_answer(pred), normalize_answer(gold)
    if p in ("yes", "no", "noanswer") and p != g:
        return 0.0
    if g in ("yes", "no", "noanswer") and p != g:
        return 0.0
    pt, gt = p.split(), g.split()
    common = Counter(pt) & Counter(gt)
    n = sum(common.values())
    if n == 0:
        return 0.0
    prec, rec = n / len(pt), n / len(gt)
    return 2 * prec * rec / (prec + rec)


def context_recall(ctx_titles, gold_titles) -> float:
    gold = set(gold_titles)
    return len(gold & set(ctx_titles)) / len(gold) if gold else 1.0


def bootstrap_ci(x, n_boot=2000, seed=0):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def paired_diff_ci(a, b, n_boot=2000, seed=0):
    """Mean(a-b) with 95% bootstrap CI (paired)."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    lo, hi = bootstrap_ci(d, n_boot, seed)
    return float(d.mean()), lo, hi
