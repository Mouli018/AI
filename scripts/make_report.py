"""Turn results/*.csv into paper-ready artefacts: figures + LaTeX tables + LaTeX macros.

    python scripts/make_report.py --results results --paper paper
Everything quoted in the paper's results/discussion comes from paper/results_macros.tex (auto-generated),
so the manuscript can never disagree with results/summary.csv.
"""
import argparse, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LABEL = {"closed_book": "Closed-book", "naive_rag": "Naive RAG (k=5)", "iter_rag": "Iterative RAG",
         "adaptive_rag": "Adaptive-RAG (no CE)", "contextrouter": "ContextRouter (ours)",
         "cr_fixed_budget": "CR w/o adaptive budget", "cr_no_critic": "CR w/o critic loop"}
ABBR = {"closed_book": "cb", "naive_rag": "nv", "iter_rag": "it", "adaptive_rag": "ar", "contextrouter": "cr",
        "cr_fixed_budget": "crfb", "cr_no_critic": "crnc"}


def pct(x):
    return "--" if pd.isna(x) else f"{100 * x:.1f}"


def num(x, d=0):
    return "--" if pd.isna(x) else f"{x:.{d}f}"


def main(res, paper):
    os.makedirs(f"{paper}/figs", exist_ok=True)
    pending = not os.path.exists(f"{res}/summary.csv")
    S = (pd.DataFrame(columns=["system", "n"]) if pending else pd.read_csv(f"{res}/summary.csv")).set_index("system")
    order = [s for s in LABEL if s in S.index]
    sw = pd.read_csv(f"{res}/summary_sweep.csv") if os.path.exists(f"{res}/summary_sweep.csv") else None
    pv = pd.read_csv(f"{res}/paired_vs_contextrouter.csv").set_index("vs") if os.path.exists(f"{res}/paired_vs_contextrouter.csv") else None
    bt = pd.read_csv(f"{res}/f1_by_type.csv").set_index("system") if os.path.exists(f"{res}/f1_by_type.csv") else None

    # ---------------- macros
    M = []
    def mac(name, val):
        M.append(f"\\newcommand{{\\{name}}}{{{val}}}")
    mac("nQ", "--" if pending else int(S["n"].max()))
    import json
    rc = json.load(open(f"{res}/run_config.json")) if os.path.exists(f"{res}/run_config.json") else {}
    mac("nPar", rc.get("paragraphs", "--")); mac("mainModel", rc.get("main_model", "--")); mac("fastModel", rc.get("fast_model", "--")); mac("retrKind", rc.get("retriever", "--"))
    for s in LABEL:
        a = ABBR[s]
        if s in S.index:
            r = S.loc[s]
            mac(f"emS{a}", pct(r["em"])); mac(f"fOne{a}", pct(r["f1"]))
            mac(f"fLo{a}", pct(r["f1_lo"])); mac(f"fHi{a}", pct(r["f1_hi"]))
            mac(f"ctxTok{a}", num(r["ctx_tokens"])); mac(f"totTok{a}", num(r["total_tokens"]))
            mac(f"calls{a}", num(r["calls"], 2)); mac(f"recall{a}", pct(r["ctx_recall"]))
            mac(f"lat{a}", num(r["latency"], 2)); mac(f"iters{a}", num(r["iterations"], 2))
            mac(f"viol{a}", num(r["violations"], 3)); mac(f"direct{a}", pct(r["direct_rate"]))
            mac(f"faith{a}", pct(r["faithful"]))
        else:
            for k in ("emS", "fOne", "fLo", "fHi", "ctxTok", "totTok", "calls", "recall", "lat", "iters", "viol", "direct", "faith"):
                mac(f"{k}{a}", "--")
    def rel(a, b, col):
        if a in S.index and b in S.index and S.loc[b, col]:
            return 100 * (S.loc[a, col] - S.loc[b, col]) / S.loc[b, col]
        return float("nan")
    def sgn(x, d=1):
        return "--" if pd.isna(x) else f"{x:+.{d}f}"
    for b, tag in (("adaptive_rag", "ar"), ("naive_rag", "nv"), ("iter_rag", "it")):
        if "contextrouter" in S.index and b in S.index:
            mac(f"dF{tag}", sgn(100 * (S.loc["contextrouter", "f1"] - S.loc[b, "f1"])))
        else:
            mac(f"dF{tag}", "--")
        mac(f"dTokPct{tag}", sgn(rel("contextrouter", b, "ctx_tokens")))
        mac(f"dTotPct{tag}", sgn(rel("contextrouter", b, "total_tokens")))
        mac(f"dCallsPct{tag}", sgn(rel("contextrouter", b, "calls")))
        if pv is not None and b in pv.index:
            r = pv.loc[b]
            mac(f"dFlo{tag}", sgn(100 * r["f1_lo"])); mac(f"dFhi{tag}", sgn(100 * r["f1_hi"]))
        else:
            mac(f"dFlo{tag}", "--"); mac(f"dFhi{tag}", "--")
    if sw is not None and len(sw):
        sw["b"] = sw["system"].str.extract(r"_(\d+)$").astype(int)
        bl = sorted(sw["b"].unique())
        mac("swLowB", int(bl[0])); mac("swMidB", int(bl[len(bl) // 2]))
        for kind, tag in (("cr_budget", "swcr"), ("raw_budget", "swraw")):
            d = sw[sw["system"].str.startswith(kind)].set_index("b")
            mac(f"{tag}Low", pct(d.loc[bl[0], "f1"]) if bl[0] in d.index else "--")
            mac(f"{tag}Mid", pct(d.loc[bl[len(bl) // 2], "f1"]) if bl[len(bl) // 2] in d.index else "--")
        crd = sw[sw["system"].str.startswith("cr_budget")].set_index("b")["f1"]
        rwd = sw[sw["system"].str.startswith("raw_budget")].set_index("b")["f1"]
        common = crd.index.intersection(rwd.index)
        if len(common):
            mac("swWins", int((crd[common] > rwd[common]).sum())); mac("swTotal", len(common))
            mac("swMeanGain", sgn(100 * float((crd[common] - rwd[common]).mean())))
    if sw is None or not len(sw):
        for k in ("swLowB", "swMidB", "swcrLow", "swcrMid", "swrawLow", "swrawMid", "swWins", "swTotal", "swMeanGain"):
            mac(k, "--")
    open(f"{paper}/results_macros.tex", "w").write("\n".join(M) + "\n")

    # ---------------- tables
    rows = []
    for s in order:
        r = S.loc[s]
        name = LABEL[s].replace("&", "\\&")
        if s == "contextrouter":
            name = "\\textbf{" + name + "}"
        rows.append(f"{name} & {pct(r['em'])} & {pct(r['f1'])} [{pct(r['f1_lo'])}, {pct(r['f1_hi'])}] & "
                    f"{num(r['ctx_tokens'])} & {num(r['calls'], 2)} & {pct(r['ctx_recall'])} & {num(r['latency'], 1)} \\\\")
    if pending:
        open(f"{paper}/results_table.tex", "w").write("\\begin{tabular}{l}\n\\hline\n(results pending: run scripts/evaluate.py then scripts/make\\_report.py)\\\\\n\\hline\n\\end{tabular}\n")
        open(f"{paper}/sweep_table.tex", "w").write("\\begin{tabular}{l}\n\\hline\n(sweep pending)\\\\\n\\hline\n\\end{tabular}\n")
        print("pending mode: wrote placeholder macros/tables (no results found)")
        return
    tab = ("\\begin{tabular}{lcccccc}\n\\hline\nSystem & EM & F1 [95\\% CI] & Ctx tok. & Calls & Ctx recall & Lat. (s) \\\\\n\\hline\n"
           + "\n".join(rows) + "\n\\hline\n\\end{tabular}\n")
    open(f"{paper}/results_table.tex", "w").write(tab)
    if sw is not None and len(sw):
        sw["kind"] = sw["system"].str.rsplit("_", n=1).str[0]
        piv = sw.pivot(index="b", columns="kind", values="f1")
        tk = sw.pivot(index="b", columns="kind", values="ctx_tokens")
        lines = [f"{int(b)} & {pct(piv.loc[b].get('raw_budget'))} & {pct(piv.loc[b].get('cr_budget'))} & "
                 f"{num(tk.loc[b].get('raw_budget'))} & {num(tk.loc[b].get('cr_budget'))} \\\\" for b in piv.index]
        open(f"{paper}/sweep_table.tex", "w").write(
            "\\begin{tabular}{rcccc}\n\\hline\nBudget $B$ & F1 truncation & F1 CE & tok. trunc. & tok. CE \\\\\n\\hline\n"
            + "\n".join(lines) + "\n\\hline\n\\end{tabular}\n")

    # ---------------- figures
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    x = np.arange(len(order))
    f1 = [100 * S.loc[s, "f1"] for s in order]
    err = [[100 * (S.loc[s, "f1"] - S.loc[s, "f1_lo"]) for s in order], [100 * (S.loc[s, "f1_hi"] - S.loc[s, "f1"]) for s in order]]
    cols = ["#C5221F" if s == "contextrouter" else "#7A9CC6" for s in order]
    ax[0].bar(x, f1, yerr=err, color=cols, capsize=3)
    ax[0].set_xticks(x); ax[0].set_xticklabels([LABEL[s].replace(" (", "\n(") for s in order], rotation=60, ha="right", fontsize=7)
    ax[0].set_ylabel("F1 (%)"); ax[0].set_title("Answer quality (95% CI)", fontsize=9)
    ax[1].bar(x, [S.loc[s, "ctx_tokens"] for s in order], color=cols)
    ax[1].set_xticks(x); ax[1].set_xticklabels([LABEL[s].replace(" (", "\n(") for s in order], rotation=60, ha="right", fontsize=7)
    ax[1].set_ylabel("Final context tokens"); ax[1].set_title("Context cost", fontsize=9)
    plt.tight_layout(); plt.savefig(f"{paper}/figs/fig_main.png", dpi=200); plt.close()

    if sw is not None and len(sw):
        fig, ax = plt.subplots(figsize=(4.6, 3.3))
        for kind, lab, c, mk in (("raw_budget", "Hard truncation", "#7A9CC6", "o"), ("cr_budget", "Context engineering", "#C5221F", "s")):
            d = sw[sw["system"].str.startswith(kind)].sort_values("b")
            ax.plot(d["b"], 100 * d["f1"], marker=mk, color=c, label=lab)
        ax.set_xlabel("Context budget B (tokens)"); ax.set_ylabel("F1 (%)")
        ax.set_title("Accuracy vs. context budget", fontsize=9); ax.legend(fontsize=7); ax.grid(alpha=.3)
        plt.tight_layout(); plt.savefig(f"{paper}/figs/fig_sweep.png", dpi=200); plt.close()
    if bt is not None:
        d = bt.loc[[s for s in order if s in bt.index]]
        d.to_csv(f"{res}/f1_by_type_ordered.csv")
    print("wrote", f"{paper}/results_macros.tex, results_table.tex, figs/*.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--paper", default="paper")
    a = ap.parse_args()
    main(a.results, a.paper)
