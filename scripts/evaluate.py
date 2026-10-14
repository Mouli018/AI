"""Run all systems on the same questions and write results/ (resumable; LLM calls are cached).

    python scripts/evaluate.py --data data/hotpot_subset.json --n 100
    python scripts/evaluate.py --data data/hotpot_subset.json --n 100 --sweep --sweep_n 60 --judge
"""
import argparse, json, os, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from contextrouter.agents import build_graph, judge_faithfulness, make_systems, run_question, sweep_systems
from contextrouter.config import get_settings
from contextrouter.data import build_corpus, gold_titles, load_questions
from contextrouter.llm import LLM
from contextrouter.metrics import bootstrap_ci, context_recall, exact_match, f1_score, paired_diff_ci
from contextrouter.retriever import Retriever

MAIN = ["closed_book", "naive_rag", "iter_rag", "adaptive_rag", "contextrouter", "cr_fixed_budget", "cr_no_critic"]


def run_system(name, cfg, questions, llm, retriever, workers, judge, store, lock, done):
    graph = build_graph(llm, retriever, cfg)

    def one(q):
        key = (name, q["_id"])
        if key in done:
            return
        try:
            r = run_question(graph, q["question"])
        except Exception as e:
            print(f"  [{name}] {q['_id']} failed: {e}")
            return
        m = r["meter"]
        row = {"system": name, "qid": q["_id"], "type": q.get("type", ""), "level": q.get("level", ""),
               "gold": q["answer"], "pred": r["answer"], "em": exact_match(r["answer"], q["answer"]),
               "f1": f1_score(r["answer"], q["answer"]), "ctx_tokens": r["ctx_tokens"],
               "ctx_recall": context_recall(r["ctx_titles"], gold_titles(q)),
               "prompt_tokens": m.prompt_tokens, "completion_tokens": m.completion_tokens,
               "calls": m.calls, "latency": round(m.latency, 3), "iterations": r["iterations"],
               "pred_qtype": r["qtype"], "violations": m.violations,
               "faithful": judge_faithfulness(llm, q["question"], r["ctx_text"], r["answer"]) if judge else None}
        with lock:
            store.write(json.dumps(row) + "\n")
            store.flush()

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(one, questions))
    print(f"  {name}: done in {time.time() - t0:.0f}s")


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for s, g in df.groupby("system", sort=False):
        f_lo, f_hi = bootstrap_ci(g["f1"])
        e_lo, e_hi = bootstrap_ci(g["em"])
        rows.append({"system": s, "n": len(g), "em": g["em"].mean(), "em_lo": e_lo, "em_hi": e_hi,
                     "f1": g["f1"].mean(), "f1_lo": f_lo, "f1_hi": f_hi,
                     "ctx_tokens": g["ctx_tokens"].mean(), "prompt_tokens": g["prompt_tokens"].mean(),
                     "total_tokens": (g["prompt_tokens"] + g["completion_tokens"]).mean(),
                     "calls": g["calls"].mean(), "latency": g["latency"].mean(),
                     "ctx_recall": g["ctx_recall"].mean(), "iterations": g["iterations"].mean(),
                     "violations": g["violations"].mean(),
                     "direct_rate": (g["pred_qtype"] == "direct").mean(),
                     "faithful": g["faithful"].dropna().mean() if g["faithful"].notna().any() else float("nan")})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/hotpot_subset.json")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--systems", nargs="*", default=MAIN)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--sweep_n", type=int, default=60)
    ap.add_argument("--budgets", nargs="*", type=int, default=[150, 300, 600, 900, 1500])
    ap.add_argument("--judge", action="store_true")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    s = get_settings()
    os.makedirs(a.out, exist_ok=True)
    questions = load_questions(a.data, a.n)
    corpus = build_corpus(questions)
    retriever = Retriever(corpus, s.retriever)
    llm = LLM(s)
    print(f"{len(questions)} questions, {len(corpus)} paragraphs, retriever={s.retriever}, mock={s.mock}")
    json.dump({"data": a.data, "n": len(questions), "paragraphs": len(corpus), "retriever": s.retriever,
               "main_model": s.main_model, "fast_model": s.fast_model, "mock": s.mock, "judge": a.judge,
               "sweep_budgets": a.budgets, "sweep_n": a.sweep_n}, open(f"{a.out}/run_config.json", "w"), indent=1)

    path = f"{a.out}/per_question.jsonl"
    done = set()
    if os.path.exists(path):
        for line in open(path):
            try:
                d = json.loads(line)
                done.add((d["system"], d["qid"]))
            except Exception:
                pass
    lock = threading.Lock()
    systems = make_systems()
    with open(path, "a") as store:
        for name in a.systems:
            print(f"[main] {name}")
            run_system(name, systems[name], questions, llm, retriever, a.workers, a.judge, store, lock, done)
        if a.sweep:
            sw = sweep_systems(a.budgets)
            for name, cfg in sw.items():
                print(f"[sweep] {name}")
                run_system(name, cfg, questions[: a.sweep_n], llm, retriever, a.workers, False, store, lock, done)

    df = pd.read_json(path, lines=True)
    df.to_csv(f"{a.out}/per_question.csv", index=False)
    main_df = df[df["system"].isin(a.systems)]
    summ = summarize(main_df)
    summ.to_csv(f"{a.out}/summary.csv", index=False)
    sw_df = df[df["system"].str.startswith(("cr_budget_", "raw_budget_"))]
    if len(sw_df):
        summarize(sw_df).to_csv(f"{a.out}/summary_sweep.csv", index=False)
    # by question type
    bt = main_df.groupby(["system", "type"])["f1"].mean().unstack().reset_index()
    bt.to_csv(f"{a.out}/f1_by_type.csv", index=False)
    # paired comparisons of the full method against every other system
    if "contextrouter" in set(main_df["system"]):
        base = main_df[main_df["system"] == "contextrouter"].set_index("qid")
        pr = []
        for sname in main_df["system"].unique():
            if sname == "contextrouter":
                continue
            o = main_df[main_df["system"] == sname].set_index("qid")
            ids = base.index.intersection(o.index)
            if len(ids) == 0:
                continue
            d, lo, hi = paired_diff_ci(base.loc[ids, "f1"], o.loc[ids, "f1"])
            dt, tlo, thi = paired_diff_ci(base.loc[ids, "ctx_tokens"], o.loc[ids, "ctx_tokens"])
            pr.append({"vs": sname, "n": len(ids), "f1_diff": d, "f1_lo": lo, "f1_hi": hi,
                       "ctx_tok_diff": dt, "ctx_tok_lo": tlo, "ctx_tok_hi": thi})
        pd.DataFrame(pr).to_csv(f"{a.out}/paired_vs_contextrouter.csv", index=False)
    pd.set_option("display.width", 200)
    print(summ[["system", "n", "em", "f1", "ctx_tokens", "calls", "ctx_recall", "iterations", "violations"]].round(3).to_string(index=False))
