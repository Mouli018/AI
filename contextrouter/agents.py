"""LangGraph orchestration of the ContextRouter agents.

START -> analyze -> plan -> (retrieve -> engineer ->) answer -> critique -> (retrieve ...)|END

All systems in the paper (baselines, full method, ablations) are the SAME graph with different
PipelineConfig flags, so measured differences are attributable to those flags only.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, TypedDict
from langgraph.graph import StateGraph, START, END

from .config import BUDGETS, TOPK
from .context_engineer import engineer_context, raw_context
from .llm import LLM, Meter, call_json
from .schemas import (AnswerDraft, Critique, EngineeredContext, Judgement, QueryPlan, StrategyDecision)

ANALYZER_SYS = (
    "ROLE=ANALYZER. You classify a question for a retrieval system. Reply with JSON: "
    '{"qtype": "direct" | "single_hop" | "multi_hop" | "comparison", "sub_questions": [..]}. '
    "direct = universally well-known fact answerable without documents (use rarely); "
    "single_hop = one fact lookup in one document; "
    "multi_hop = needs chaining facts across documents (give up to 3 ordered sub_questions); "
    "comparison = compares two entities (one sub_question per entity). "
    "When unsure choose multi_hop."
)
ANSWERER_SYS = (
    "ROLE=ANSWERER. Answer the question using ONLY the context. Reply with JSON "
    '{"answer": "<short phrase, at most 6 words; exactly yes or no for yes/no questions>", "confidence": <0-1>}. '
    "If the context is insufficient, give your best guess."
)
CLOSED_SYS = (
    "ROLE=CLOSEDBOOK. Answer from your own knowledge. Reply with JSON "
    '{"answer": "<short phrase, at most 6 words; exactly yes or no for yes/no questions>", "confidence": <0-1>}.'
)
CRITIC_SYS = (
    "ROLE=CRITIC. Given a question, a context and a draft answer, decide (a) whether the answer is supported by "
    "the context and (b) whether the context contains the evidence for EVERY reasoning hop. Reply with JSON "
    '{"grounded": true|false, "sufficient": true|false, "followup_query": "<search query for the missing fact, '
    'or empty string>"}.'
)
JUDGE_SYS = (
    "ROLE=JUDGE. Is the answer fully supported by the context (not just plausible)? "
    'Reply with JSON {"supported": true|false}.'
)


@dataclass
class PipelineConfig:
    name: str
    retrieval: bool = True
    use_analyzer: bool = True      # adaptive routing by question type
    use_ce: bool = True            # context engineering (else raw top-k paragraphs)
    adaptive_budget: bool = True   # budget/k chosen per query type (else fixed_*)
    fixed_budget: int = 900
    fixed_k: int = 5
    use_critic: bool = True        # evaluator-optimizer loop
    max_iters: int = 3
    decompose: bool = True         # use analyzer sub-questions as extra queries
    raw_truncate: bool = False     # truncate raw context to fixed_budget (sweep baseline)


class State(TypedDict, total=False):
    question: str
    meter: Any
    plan: QueryPlan
    decision: StrategyDecision
    queries: list
    queries_all: list
    pool: dict
    context: EngineeredContext
    draft: AnswerDraft
    critiques: list
    iteration: int
    done: bool


def build_graph(llm: LLM, retriever, cfg: PipelineConfig):
    def analyze(state: State):
        q = state["question"]
        if cfg.use_analyzer and cfg.retrieval:
            plan = call_json(llm, state["meter"], QueryPlan, ANALYZER_SYS, f"Question: {q}", "fast", 200) or QueryPlan()
        else:
            plan = QueryPlan(qtype="multi_hop" if cfg.retrieval else "direct")
        return {"plan": plan, "iteration": 0, "critiques": [], "pool": {}}

    def plan_strategy(state: State):
        p, q = state["plan"], state["question"]
        if cfg.use_analyzer and cfg.adaptive_budget:
            budget, k = BUDGETS[p.qtype], TOPK[p.qtype]
        else:
            budget, k = cfg.fixed_budget, cfg.fixed_k
        queries = [q]
        if cfg.decompose and cfg.use_analyzer and p.qtype in ("multi_hop", "comparison"):
            queries += [s for s in p.sub_questions if s.strip() and s.strip() != q][:3]
        return {"decision": StrategyDecision(budget=budget, k=k, queries=queries),
                "queries": queries, "queries_all": list(queries)}

    def route_plan(state: State):
        return "retrieve" if state["plan"].needs_retrieval else "answer"

    def retrieve(state: State):
        d = state["decision"]
        pool = dict(state["pool"])
        for i, qq in enumerate(state["queries"]):
            kk = d.k if i == 0 else max(2, d.k // 2)
            for c in retriever.search(qq, kk):
                if c.title not in pool or c.score > pool[c.title].score:
                    pool[c.title] = c
        return {"pool": pool}

    def engineer(state: State):
        d = state["decision"]
        chunks = list(state["pool"].values())
        if cfg.use_ce:
            ctx = engineer_context(chunks, state["queries_all"], d.budget, retriever)
        else:
            ctx = raw_context(chunks, d.k + 2 * state["iteration"], d.budget if cfg.raw_truncate else None)
        return {"context": ctx}

    def answer(state: State):
        q, ctx = state["question"], state.get("context")
        if ctx is not None and ctx.text:
            sys_, user = ANSWERER_SYS, f"CONTEXT:\n{ctx.text}\n\nQUESTION: {q}"
        else:
            sys_, user = CLOSED_SYS, f"QUESTION: {q}"
        d = call_json(llm, state["meter"], AnswerDraft, sys_, user, "main", 60) or AnswerDraft(answer="")
        return {"draft": d}

    def critique(state: State):
        it = state["iteration"]
        if not (cfg.use_critic and state["plan"].needs_retrieval and it < cfg.max_iters):
            return {"done": True}
        user = (f"CONTEXT:\n{state['context'].text}\n\nDRAFT ANSWER: {state['draft'].answer}\n\n"
                f"QUESTION: {state['question']}")
        c = call_json(llm, state["meter"], Critique, CRITIC_SYS, user, "fast", 120)
        crits = state["critiques"] + [c.model_dump() if c else None]
        fq = c.followup_query.strip() if c else ""
        if c is None or (c.grounded and c.sufficient) or not fq:
            return {"done": True, "critiques": crits}
        return {"done": False, "critiques": crits, "iteration": it + 1,
                "queries": [fq], "queries_all": state["queries_all"] + [fq]}

    def route_critique(state: State):
        return END if state.get("done") else "retrieve"

    g = StateGraph(State)
    g.add_node("analyze", analyze)
    g.add_node("plan", plan_strategy)
    g.add_node("retrieve", retrieve)
    g.add_node("engineer", engineer)
    g.add_node("answer", answer)
    g.add_node("critique", critique)
    g.add_edge(START, "analyze")
    g.add_edge("analyze", "plan")
    g.add_conditional_edges("plan", route_plan, {"retrieve": "retrieve", "answer": "answer"})
    g.add_edge("retrieve", "engineer")
    g.add_edge("engineer", "answer")
    g.add_edge("answer", "critique")
    g.add_conditional_edges("critique", route_critique, {"retrieve": "retrieve", END: END})
    return g.compile()


def run_question(graph, question: str) -> dict:
    meter = Meter()
    out = graph.invoke({"question": question, "meter": meter}, {"recursion_limit": 60})
    ctx = out.get("context") or EngineeredContext()
    return {
        "answer": out["draft"].answer,
        "qtype": out["plan"].qtype,
        "iterations": out.get("iteration", 0),
        "queries": out.get("queries_all", []),
        "ctx_titles": ctx.titles,
        "ctx_tokens": ctx.tokens,
        "ctx_text": ctx.text,
        "critiques": out.get("critiques", []),
        "meter": meter,
    }


def judge_faithfulness(llm: LLM, question: str, ctx_text: str, answer: str) -> float | None:
    """Optional LLM-as-judge faithfulness (kept in its own Meter, not counted in system cost)."""
    if not ctx_text:
        return None
    m = Meter()
    j = call_json(llm, m, Judgement, JUDGE_SYS,
                  f"CONTEXT:\n{ctx_text}\n\nQUESTION: {question}\nANSWER: {answer}", "fast", 40)
    return None if j is None else float(j.supported)


def make_systems() -> dict[str, PipelineConfig]:
    return {
        "closed_book":     PipelineConfig("closed_book", retrieval=False, use_analyzer=False, use_ce=False, use_critic=False),
        "naive_rag":       PipelineConfig("naive_rag", use_analyzer=False, use_ce=False, adaptive_budget=False, fixed_k=5, use_critic=False),
        "iter_rag":        PipelineConfig("iter_rag", use_analyzer=False, use_ce=False, adaptive_budget=False, fixed_k=5, use_critic=True),
        "adaptive_rag":    PipelineConfig("adaptive_rag", use_analyzer=True, use_ce=False, adaptive_budget=True, use_critic=True),
        "contextrouter":   PipelineConfig("contextrouter", use_analyzer=True, use_ce=True, adaptive_budget=True, use_critic=True),
        # ablations of the full method
        "cr_fixed_budget": PipelineConfig("cr_fixed_budget", use_analyzer=True, use_ce=True, adaptive_budget=False, fixed_budget=900, fixed_k=8, use_critic=True),
        "cr_no_critic":    PipelineConfig("cr_no_critic", use_analyzer=True, use_ce=True, adaptive_budget=True, use_critic=False),
    }


def sweep_systems(budgets: list[int]) -> dict[str, PipelineConfig]:
    """Accuracy-vs-context-budget sweep: context engineering vs. hard truncation at the SAME budget."""
    out = {}
    for b in budgets:
        out[f"cr_budget_{b}"] = PipelineConfig(f"cr_budget_{b}", use_analyzer=False, use_ce=True, adaptive_budget=False,
                                               fixed_budget=b, fixed_k=12, use_critic=False)
        out[f"raw_budget_{b}"] = PipelineConfig(f"raw_budget_{b}", use_analyzer=False, use_ce=False, adaptive_budget=False,
                                                fixed_budget=b, fixed_k=12, use_critic=False, raw_truncate=True)
    return out
