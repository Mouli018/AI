"""Download HotpotQA (distractor setting, validation split) and save a reproducible random subset.

    python scripts/prepare_data.py --n 300 --seed 42
-> data/hotpot_subset.json   (original HotpotQA JSON format)
-> data/DATA_CARD.json       (source, split, n, seed, ids) for the DATASET.md / paper
"""
import argparse, json, random, urllib.request
from pathlib import Path

URL = "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json"


def from_hf():
    from datasets import load_dataset
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    out = []
    for r in ds:
        ctx = [[t, list(s)] for t, s in zip(r["context"]["title"], r["context"]["sentences"])]
        sf = [[t, int(i)] for t, i in zip(r["supporting_facts"]["title"], r["supporting_facts"]["sent_id"])]
        out.append({"_id": r["id"], "question": r["question"], "answer": r["answer"], "type": r["type"],
                    "level": r["level"], "supporting_facts": sf, "context": ctx})
    return out, "hf:hotpotqa/hotpot_qa (distractor, validation)"


def from_url():
    raw = urllib.request.urlopen(URL, timeout=120).read()
    return json.loads(raw), URL


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/hotpot_subset.json")
    a = ap.parse_args()
    try:
        data, src = from_hf()
    except Exception as e:
        print("HF download failed (%s); falling back to the original HotpotQA JSON URL" % e)
        data, src = from_url()
    random.Random(a.seed).shuffle(data)
    sub = data[: a.n]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(sub, open(a.out, "w"))
    json.dump({"source": src, "split": "validation (distractor)", "n": len(sub), "seed": a.seed,
               "ids": [q["_id"] for q in sub]}, open("data/DATA_CARD.json", "w"), indent=1)
    types = {}
    for q in sub:
        types[q["type"]] = types.get(q["type"], 0) + 1
    print(f"saved {len(sub)} questions to {a.out}; types={types}")
