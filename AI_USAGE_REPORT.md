# AI Tools Usage Report

Project: ContextRouter (22MDCEL11 - AI Systems Engineering for Agentic Workflows Laboratory, CAT II)
Author: Moulidharan S

## 1. AI tools used in *creating* this submission
| Tool | Used for | Where | Human oversight |
|---|---|---|---|
| Claude (Anthropic, claude.ai chat) | Project ideation and selection among alternatives; designing the experiment | Planning | Author chose the topic and approved the design |
| Claude (Anthropic, claude.ai chat) | First drafts of all source code (`contextrouter/`, `scripts/`, `app.py`), the architecture figure script, README and dataset note | Code, docs | Code was executed end to end in an offline mock-LLM smoke test and the demo API was exercised; author must run the real experiment and review the code |
| Claude (Anthropic, claude.ai chat) | First draft of the IEEE LaTeX manuscript and BibTeX file; the manuscript quotes numbers through auto-generated macros (`paper/results_macros.tex`) rather than typed values | Paper | Author reviews the text against the final results and checks every reference before submission |

## 2. AI models used as *components of the system* (runtime, not authorship)
- Answerer: `llama-3.3-70b-versatile`; Analyzer / Critic: `llama-3.1-8b-instant`, accessed through an OpenAI-compatible API (default Groq). Their outputs are the system under evaluation, cached in `results/llm_cache.jsonl`.

## 3. What was *not* AI-generated
- The experimental results: they are produced by `scripts/evaluate.py` and stored in `results/`.
- Dataset: public HotpotQA; nothing was synthesised for the reported experiments (the synthetic `sample_mini.json` is only a smoke-test fixture).

## 4. Verification performed
- Offline smoke test of the full pipeline (all 7 systems, budget sweep, judge path, report generation) with a deterministic mock LLM.
- Unit checks of the context engineer (token budget respected; iterative critic loop accumulates queries).
- The LaTeX manuscript compiles with the IEEEtran class.
- Reference list entries correspond to published works; the author should re-verify titles, venues and years against the publishers' pages.

## 5. Declaration
The author is responsible for the content of the submission. If any other AI tool is used before submission (e.g. for editing the paper), add it to Section 1.
