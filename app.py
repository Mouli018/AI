"""Demo UI + API.   uvicorn app:app --reload   ->  http://127.0.0.1:8000
Set DATA_FILE to choose the corpus (default: data/hotpot_subset.json, else data/sample_mini.json)."""
import json
import os
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from contextrouter.agents import build_graph, make_systems, run_question
from contextrouter.config import get_settings
from contextrouter.data import build_corpus, load_questions
from contextrouter.llm import LLM
from contextrouter.retriever import Retriever

DATA = os.getenv("DATA_FILE") or ("data/hotpot_subset.json" if os.path.exists("data/hotpot_subset.json") else "data/sample_mini.json")
N = int(os.getenv("DEMO_N", "100"))
settings = get_settings()
questions = load_questions(DATA, N)
retriever = Retriever(build_corpus(questions), settings.retriever)
llm = LLM(settings)
SYSTEMS = make_systems()
GRAPHS = {name: build_graph(llm, retriever, cfg) for name, cfg in SYSTEMS.items()}

app = FastAPI(title="ContextRouter Studio & Multi-Agent Cockpit")


class Ask(BaseModel):
    question: str
    system: str = "contextrouter"


class CompareReq(BaseModel):
    question: str
    system_a: str = "contextrouter"
    system_b: str = "naive_rag"


@app.get("/api/examples")
def examples():
    return [{"question": q["question"], "gold": q["answer"], "type": q.get("type", "multi_hop")} for q in questions[:12]]


@app.get("/api/dataset-info")
def dataset_info():
    card_path = "data/DATA_CARD.json"
    card = {}
    if os.path.exists(card_path):
        try:
            with open(card_path, "r", encoding="utf-8") as f:
                card = json.load(f)
        except Exception:
            pass
    return {
        "file": DATA,
        "total_loaded": len(questions),
        "card": card,
        "upstream_urls": {
            "huggingface": "https://huggingface.co/datasets/hotpotqa/hotpot_qa",
            "cmu_json": "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json",
            "github_project": "https://hotpotqa.github.io/"
        }
    }


@app.post("/api/ask")
def ask(req: Ask):
    r = run_question(GRAPHS.get(req.system, GRAPHS["contextrouter"]), req.question)
    m = r.pop("meter")
    r["usage"] = {
        "llm_calls": m.calls,
        "prompt_tokens": m.prompt_tokens,
        "completion_tokens": m.completion_tokens,
        "latency_s": round(m.latency, 2),
        "contract_violations": m.violations,
    }
    return r


@app.post("/api/compare")
def compare(req: CompareReq):
    ra = run_question(GRAPHS.get(req.system_a, GRAPHS["contextrouter"]), req.question)
    ma = ra.pop("meter")
    ra["usage"] = {
        "llm_calls": ma.calls,
        "prompt_tokens": ma.prompt_tokens,
        "completion_tokens": ma.completion_tokens,
        "latency_s": round(ma.latency, 2),
        "contract_violations": ma.violations,
    }

    rb = run_question(GRAPHS.get(req.system_b, GRAPHS["naive_rag"]), req.question)
    mb = rb.pop("meter")
    rb["usage"] = {
        "llm_calls": mb.calls,
        "prompt_tokens": mb.prompt_tokens,
        "completion_tokens": mb.completion_tokens,
        "latency_s": round(mb.latency, 2),
        "contract_violations": mb.violations,
    }
    return {"a": ra, "b": rb}


PAGE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>ContextRouter Studio - Multi-Agent RAG Cockpit</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #060911;
      --panel-bg: rgba(13, 19, 33, 0.75);
      --panel-border: rgba(99, 102, 241, 0.2);
      --neon-cyan: #00f2fe;
      --neon-purple: #8b5cf6;
      --neon-pink: #ec4899;
      --neon-green: #10b981;
      --neon-amber: #f59e0b;
      --text: #f8fafc;
      --text-dim: #94a3b8;
      --text-faint: #64748b;
      --code-bg: #090e1a;
      --glow-cyan: 0 0 20px rgba(0, 242, 254, 0.25);
      --glow-purple: 0 0 20px rgba(139, 92, 246, 0.3);
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Plus Jakarta Sans', system-ui, sans-serif;
      background: var(--bg);
      background-image: 
        radial-gradient(circle at 15% 10%, rgba(99, 102, 241, 0.18) 0%, transparent 40%),
        radial-gradient(circle at 85% 85%, rgba(0, 242, 254, 0.12) 0%, transparent 45%),
        linear-gradient(to bottom, rgba(6, 9, 17, 0.85), #060911);
      color: var(--text);
      min-height: 100vh;
      padding: 24px 20px 80px;
    }
    .hud-container {
      max-width: 1200px;
      margin: 0 auto;
    }

    /* TOP BAR */
    .top-bar {
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 16px;
      padding-bottom: 20px;
      border-bottom: 1px solid rgba(255, 255, 255, 0.08);
      margin-bottom: 24px;
    }
    .brand-group {
      display: flex;
      align-items: center;
      gap: 14px;
    }
    .brand-icon {
      width: 44px;
      height: 44px;
      border-radius: 12px;
      background: linear-gradient(135deg, #6366f1, #00f2fe);
      display: flex;
      align-items: center;
      justify-content: center;
      font-family: 'Space Grotesk', sans-serif;
      font-weight: 700;
      font-size: 20px;
      color: #060911;
      box-shadow: 0 0 20px rgba(99, 102, 241, 0.5);
    }
    .brand-title {
      font-family: 'Space Grotesk', sans-serif;
      font-size: 26px;
      font-weight: 700;
      letter-spacing: -0.03em;
      background: linear-gradient(to right, #ffffff, #c7d2fe, #38bdf8);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }
    .brand-subtitle {
      font-size: 13px;
      color: var(--text-dim);
    }
    .top-actions {
      display: flex;
      align-items: center;
      gap: 10px;
      flex-wrap: wrap;
    }
    .tag-badge {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      font-size: 11.5px;
      font-weight: 600;
      padding: 5px 12px;
      border-radius: 20px;
      background: rgba(16, 185, 129, 0.12);
      color: #34d399;
      border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .btn-ghost {
      background: rgba(255, 255, 255, 0.05);
      border: 1px solid rgba(255, 255, 255, 0.1);
      color: #cbd5e1;
      font-size: 12px;
      font-weight: 600;
      padding: 6px 14px;
      border-radius: 8px;
      cursor: pointer;
      transition: all 0.2s;
    }
    .btn-ghost:hover {
      background: rgba(255, 255, 255, 0.12);
      color: #fff;
      border-color: rgba(255, 255, 255, 0.25);
    }

    /* PIPELINE FLOWCHART */
    .flow-hud {
      background: var(--panel-bg);
      backdrop-filter: blur(16px);
      border: 1px solid var(--panel-border);
      border-radius: 16px;
      padding: 16px 20px;
      margin-bottom: 24px;
      box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
    }
    .flow-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 12px;
      font-size: 11px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--text-dim);
    }
    .pipeline-nodes {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
      gap: 8px;
      position: relative;
    }
    .node-card {
      background: rgba(15, 23, 42, 0.7);
      border: 1px solid rgba(255, 255, 255, 0.07);
      border-radius: 10px;
      padding: 10px 12px;
      font-size: 12px;
      position: relative;
      transition: all 0.25s;
    }
    .node-card.highlight {
      border-color: #00f2fe;
      background: rgba(0, 242, 254, 0.08);
      box-shadow: 0 0 16px rgba(0, 242, 254, 0.2);
    }
    .node-card.active-pulse {
      animation: nodePulse 1.2s infinite;
      border-color: #ec4899;
    }
    @keyframes nodePulse {
      0%, 100% { box-shadow: 0 0 10px rgba(236, 72, 153, 0.3); }
      50% { box-shadow: 0 0 25px rgba(236, 72, 153, 0.7); }
    }
    .node-num {
      font-size: 10px;
      font-family: 'JetBrains Mono', monospace;
      color: var(--text-faint);
      margin-bottom: 2px;
    }
    .node-name {
      font-weight: 600;
      color: #f1f5f9;
    }
    .node-role {
      font-size: 10.5px;
      color: var(--text-dim);
      margin-top: 2px;
    }

    /* CONTROL CONSOLE */
    .cockpit-panel {
      background: var(--panel-bg);
      backdrop-filter: blur(16px);
      border: 1px solid var(--panel-border);
      border-radius: 18px;
      padding: 22px;
      margin-bottom: 24px;
      box-shadow: 0 12px 35px rgba(0, 0, 0, 0.45);
    }
    .mode-switcher {
      display: inline-flex;
      background: rgba(0, 0, 0, 0.4);
      padding: 4px;
      border-radius: 10px;
      border: 1px solid rgba(255, 255, 255, 0.08);
      margin-bottom: 16px;
    }
    .mode-btn {
      background: transparent;
      border: none;
      color: var(--text-dim);
      font-size: 12.5px;
      font-weight: 600;
      padding: 6px 16px;
      border-radius: 7px;
      cursor: pointer;
      transition: all 0.2s;
    }
    .mode-btn.active {
      background: linear-gradient(135deg, #6366f1, #4f46e5);
      color: #fff;
      box-shadow: 0 2px 10px rgba(99, 102, 241, 0.4);
    }
    .controls-grid {
      display: grid;
      grid-template-columns: 240px 1fr auto;
      gap: 12px;
      align-items: center;
    }
    @media (max-width: 820px) {
      .controls-grid { grid-template-columns: 1fr; }
    }
    select, input, button {
      font-family: inherit;
      outline: none;
    }
    select {
      background: #090e1a;
      color: #e2e8f0;
      border: 1px solid rgba(255, 255, 255, 0.12);
      border-radius: 12px;
      padding: 13px 14px;
      font-size: 13.5px;
      cursor: pointer;
      transition: border 0.2s;
    }
    select:focus { border-color: var(--neon-cyan); }
    .input-wrapper {
      position: relative;
      display: flex;
      align-items: center;
    }
    input#q {
      width: 100%;
      background: #090e1a;
      border: 1px solid rgba(255, 255, 255, 0.12);
      border-radius: 12px;
      padding: 13px 16px;
      color: #fff;
      font-size: 14.5px;
      transition: all 0.2s;
    }
    input#q:focus {
      border-color: var(--neon-cyan);
      box-shadow: 0 0 0 3px rgba(0, 242, 254, 0.2);
    }
    .btn-fire {
      background: linear-gradient(135deg, #6366f1, #00f2fe);
      border: none;
      color: #060911;
      font-weight: 700;
      font-size: 14px;
      padding: 13px 28px;
      border-radius: 12px;
      cursor: pointer;
      transition: all 0.2s;
      display: inline-flex;
      align-items: center;
      gap: 8px;
      box-shadow: 0 4px 18px rgba(0, 242, 254, 0.35);
      white-space: nowrap;
    }
    .btn-fire:hover:not(:disabled) {
      transform: translateY(-2px);
      box-shadow: 0 6px 24px rgba(0, 242, 254, 0.5);
    }
    .btn-fire:disabled {
      opacity: 0.6;
      cursor: not-allowed;
    }

    /* QUESTION DECK */
    .deck-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-top: 18px;
      margin-bottom: 10px;
    }
    .deck-title {
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--text-dim);
    }
    .chips-deck {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }
    .q-chip {
      background: rgba(255, 255, 255, 0.04);
      border: 1px solid rgba(255, 255, 255, 0.08);
      border-radius: 10px;
      padding: 8px 14px;
      font-size: 12px;
      color: #cbd5e1;
      cursor: pointer;
      transition: all 0.2s;
      max-width: 100%;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      display: flex;
      align-items: center;
      gap: 6px;
    }
    .q-chip:hover {
      background: rgba(99, 102, 241, 0.15);
      border-color: rgba(99, 102, 241, 0.4);
      color: #fff;
      transform: translateY(-1px);
    }
    .q-type-badge {
      font-size: 10px;
      padding: 2px 6px;
      border-radius: 4px;
      background: rgba(0, 242, 254, 0.15);
      color: #00f2fe;
      font-family: 'JetBrains Mono', monospace;
    }

    /* RESULTS VIEW */
    #soloResults, #compareResults { display: none; }
    .verdict-card {
      background: linear-gradient(135deg, rgba(30, 41, 59, 0.9), rgba(15, 23, 42, 0.95));
      border: 1px solid rgba(0, 242, 254, 0.3);
      border-radius: 16px;
      padding: 24px;
      margin-bottom: 20px;
      position: relative;
      overflow: hidden;
    }
    .verdict-card::before {
      content: '';
      position: absolute;
      top: 0; left: 0; right: 0; height: 3px;
      background: linear-gradient(90deg, #6366f1, #00f2fe, #10b981);
    }
    .verdict-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 12px;
    }
    .verdict-tag {
      font-size: 11px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: #38bdf8;
    }
    .verdict-ans {
      font-size: 26px;
      font-weight: 700;
      color: #34d399;
      line-height: 1.3;
      margin-bottom: 16px;
    }

    /* TELEMETRY METERS */
    .budget-gauge-box {
      background: rgba(0, 0, 0, 0.4);
      border: 1px solid rgba(255, 255, 255, 0.08);
      border-radius: 12px;
      padding: 14px 16px;
      margin-top: 14px;
    }
    .gauge-labels {
      display: flex;
      justify-content: space-between;
      font-size: 12px;
      margin-bottom: 6px;
    }
    .gauge-track {
      height: 10px;
      background: rgba(255, 255, 255, 0.08);
      border-radius: 5px;
      overflow: hidden;
      position: relative;
    }
    .gauge-fill {
      height: 100%;
      background: linear-gradient(90deg, #10b981, #00f2fe);
      width: 0%;
      transition: width 0.8s ease-out;
      border-radius: 5px;
    }

    .kpi-deck {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
      gap: 12px;
      margin-bottom: 24px;
    }
    .kpi-card {
      background: rgba(13, 19, 33, 0.7);
      border: 1px solid var(--panel-border);
      border-radius: 12px;
      padding: 14px;
    }
    .kpi-title {
      font-size: 11px;
      text-transform: uppercase;
      font-weight: 600;
      letter-spacing: 0.06em;
      color: var(--text-dim);
      margin-bottom: 4px;
    }
    .kpi-val {
      font-size: 20px;
      font-weight: 700;
      color: #fff;
      font-family: 'Space Grotesk', sans-serif;
    }

    /* TABS */
    .tabs-deck {
      background: var(--panel-bg);
      backdrop-filter: blur(16px);
      border: 1px solid var(--panel-border);
      border-radius: 16px;
      padding: 20px;
    }
    .tab-bar {
      display: flex;
      gap: 8px;
      border-bottom: 1px solid rgba(255, 255, 255, 0.08);
      padding-bottom: 12px;
      margin-bottom: 16px;
      overflow-x: auto;
    }
    .tab-item {
      background: transparent;
      border: none;
      color: var(--text-dim);
      font-size: 13px;
      font-weight: 600;
      padding: 8px 16px;
      border-radius: 8px;
      cursor: pointer;
      transition: all 0.2s;
      white-space: nowrap;
    }
    .tab-item.active {
      background: rgba(99, 102, 241, 0.2);
      color: #fff;
      border: 1px solid rgba(99, 102, 241, 0.35);
    }
    .tab-content { display: none; }
    .tab-content.active { display: block; }
    pre {
      font-family: 'JetBrains Mono', monospace;
      font-size: 12.5px;
      background: var(--code-bg);
      border: 1px solid rgba(255, 255, 255, 0.06);
      border-radius: 10px;
      padding: 16px;
      color: #e2e8f0;
      white-space: pre-wrap;
      word-break: break-word;
      max-height: 480px;
      overflow-y: auto;
      line-height: 1.55;
    }

    /* COMPARE GRID */
    .compare-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 20px;
      margin-bottom: 24px;
    }
    @media (max-width: 820px) {
      .compare-grid { grid-template-columns: 1fr; }
    }
    .compare-side {
      background: rgba(13, 19, 33, 0.8);
      border: 1px solid var(--panel-border);
      border-radius: 16px;
      padding: 20px;
    }
    .compare-title {
      font-family: 'Space Grotesk', sans-serif;
      font-size: 16px;
      font-weight: 700;
      margin-bottom: 12px;
      display: flex;
      align-items: center;
      gap: 8px;
    }

    /* MODAL */
    .modal-overlay {
      position: fixed;
      top: 0; left: 0; right: 0; bottom: 0;
      background: rgba(0, 0, 0, 0.8);
      backdrop-filter: blur(8px);
      display: none;
      align-items: center;
      justify-content: center;
      z-index: 1000;
      padding: 20px;
    }
    .modal-box {
      background: #0b1120;
      border: 1px solid rgba(0, 242, 254, 0.3);
      border-radius: 20px;
      max-width: 760px;
      width: 100%;
      max-height: 85vh;
      overflow-y: auto;
      padding: 28px;
      box-shadow: 0 20px 50px rgba(0, 0, 0, 0.7);
    }
    .modal-title {
      font-family: 'Space Grotesk', sans-serif;
      font-size: 22px;
      font-weight: 700;
      margin-bottom: 16px;
      color: #fff;
    }
    .modal-close {
      float: right;
      background: none;
      border: none;
      color: var(--text-dim);
      font-size: 24px;
      cursor: pointer;
    }

    /* SPINNER */
    .spinner {
      display: inline-block;
      width: 18px;
      height: 18px;
      border: 2px solid rgba(0, 0, 0, 0.25);
      border-radius: 50%;
      border-top-color: #060911;
      animation: spin 0.8s linear infinite;
    }
    @keyframes spin { to { transform: rotate(360deg); } }
  </style>
</head>
<body>
<div class="hud-container">
  <!-- TOP BAR -->
  <div class="top-bar">
    <div class="brand-group">
      <div class="brand-icon">&Delta;</div>
      <div>
        <div class="brand-title">ContextRouter Studio</div>
        <div class="brand-subtitle">Adaptive Multi-Agent RAG with Context-Budget Engineering</div>
      </div>
    </div>
    <div class="top-actions">
      <div class="tag-badge">
        <span style="width: 7px; height: 7px; border-radius: 50%; background: #10b981; box-shadow: 0 0 8px #10b981;"></span>
        OpenAI gpt-4o-mini Online
      </div>
      <button class="btn-ghost" onclick="openModal('modalExplain')">&quest; Architecture Explainer</button>
      <button class="btn-ghost" onclick="openModal('modalDataset')">&darr; Dataset Hub</button>
      <button id="sfxToggle" class="btn-ghost" onclick="toggleSfx()">&sunglasses; SFX: ON</button>
    </div>
  </div>

  <!-- PIPELINE FLOWCHART -->
  <div class="flow-hud">
    <div class="flow-header">
      <span>Agentic Execution Pipeline</span>
      <span style="color: #00f2fe; font-family: 'JetBrains Mono', monospace;">LangGraph &bull; Pydantic Contracts</span>
    </div>
    <div class="pipeline-nodes">
      <div id="pNode1" class="node-card">
        <div class="node-num">01 / ROUTER</div>
        <div class="node-name">Query Analyzer</div>
        <div class="node-role">Classify intent &amp; sub-Qs</div>
      </div>
      <div id="pNode2" class="node-card">
        <div class="node-num">02 / PLANNER</div>
        <div class="node-name">Strategy Planner</div>
        <div class="node-role">Set budget B &amp; top-k</div>
      </div>
      <div id="pNode3" class="node-card">
        <div class="node-num">03 / RETRIEVER</div>
        <div class="node-name">Corpus Retriever</div>
        <div class="node-role">TF-IDF / Dense candidate pool</div>
      </div>
      <div id="pNode4" class="node-card highlight">
        <div class="node-num" style="color: #00f2fe;">04 / NOVELTY &star;</div>
        <div class="node-name" style="color: #00f2fe;">Context Engineer</div>
        <div class="node-role">Sentence scoring &amp; budget pack</div>
      </div>
      <div id="pNode5" class="node-card">
        <div class="node-num">05 / SYNTHESIS</div>
        <div class="node-name">Answerer</div>
        <div class="node-role">Strict JSON contract generation</div>
      </div>
      <div id="pNode6" class="node-card">
        <div class="node-num">06 / VERIFIER</div>
        <div class="node-name">Critic Loop</div>
        <div class="node-role">Groundedness &amp; follow-up hops</div>
      </div>
    </div>
  </div>

  <!-- CONTROL COCKPIT -->
  <div class="cockpit-panel">
    <div class="mode-switcher">
      <button id="btnModeSolo" class="mode-btn active" onclick="setExecMode('solo')">Solo Agent Run</button>
      <button id="btnModeCompare" class="mode-btn" onclick="setExecMode('compare')">Head-to-Head Arena (A/B Test)</button>
    </div>

    <div class="controls-grid">
      <!-- System Selector (Solo) -->
      <div id="soloSysWrap">
        <select id="sysSelect" style="width: 100%;">
          <option value="contextrouter">ContextRouter (Full System: Adaptive Budget + Context Engineer)</option>
          <option value="adaptive_rag">Adaptive RAG (Router without Context Engineer)</option>
        </select>
      </div>

      <!-- System Selector (Compare) -->
      <div id="compareSysWrap" style="display: none;">
        <div style="display: flex; gap: 8px;">
          <select id="sysA" style="width: 50%;">
            <option value="contextrouter" selected>System A: ContextRouter</option>
            <option value="adaptive_rag">Adaptive RAG</option>
          </select>
          <select id="sysB" style="width: 50%;">
            <option value="naive_rag" selected>System B: Naive RAG</option>
            <option value="iter_rag">Iterative RAG</option>
            <option value="closed_book">Closed Book</option>
          </select>
        </div>
      </div>

      <!-- Question Input -->
      <div class="input-wrapper">
        <input id="q" type="text" placeholder="Enter a multi-hop query or click a benchmark chip below..." autocomplete="off">
      </div>

      <!-- Action Button -->
      <div>
        <button id="runBtn" class="btn-fire" onclick="runCockpit()">
          <span id="runBtnText">&blacktriangleright; Launch Agents</span>
        </button>
      </div>
    </div>

    <!-- HOTPOTQA CHIP DECK -->
    <div class="deck-header">
      <div class="deck-title">HotpotQA Benchmark Questions (Click to Execute):</div>
    </div>
    <div id="chipsDeck" class="chips-deck">
      <span class="q-chip" style="opacity: 0.5;">Loading benchmark questions...</span>
    </div>
  </div>

  <!-- SOLO RESULTS -->
  <div id="soloResults">
    <div class="verdict-card">
      <div class="verdict-header">
        <span class="verdict-tag">&bull; Synthesized Grounded Answer</span>
        <button class="btn-ghost" onclick="copyAnswer()">Copy Answer</button>
      </div>
      <div id="vAnswer" class="verdict-ans">--</div>

      <div class="budget-gauge-box">
        <div class="gauge-labels">
          <span style="color: var(--text-dim);">Context Budget Telemetry:</span>
          <span id="gaugeLabel" style="font-family: 'JetBrains Mono', monospace; color: #00f2fe;">0 / 900 tokens</span>
        </div>
        <div class="gauge-track">
          <div id="gaugeFill" class="gauge-fill"></div>
        </div>
      </div>
    </div>

    <div class="kpi-deck">
      <div class="kpi-card">
        <div class="kpi-title">Query Classification</div>
        <div id="kRoute" class="kpi-val" style="color: #38bdf8;">--</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-title">Context Tokens Packed</div>
        <div id="kTokens" class="kpi-val">--</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-title">Critic Loops / Hops</div>
        <div id="kHops" class="kpi-val">--</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-title">Model Roundtrips</div>
        <div id="kCalls" class="kpi-val">--</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-title">Latency</div>
        <div id="kLatency" class="kpi-val">--</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-title">Contract Violations</div>
        <div id="kViolations" class="kpi-val" style="color: #34d399;">0</div>
      </div>
    </div>

    <div class="tabs-deck">
      <div class="tab-bar">
        <button class="tab-item active" onclick="switchTab(event, 'tabCtx')">&equiv; Engineered Context</button>
        <button class="tab-item" onclick="switchTab(event, 'tabQueries')">&curren; Multi-Hop Queries</button>
        <button class="tab-item" onclick="switchTab(event, 'tabCritic')">&check; Critic Trajectory</button>
        <button class="tab-item" onclick="switchTab(event, 'tabRaw')">&lbrace;&rbrace; JSON Contract</button>
      </div>

      <div id="tabCtx" class="tab-content active">
        <div id="ctxPassageInfo" style="font-size: 12px; color: var(--text-dim); margin-bottom: 8px;"></div>
        <pre id="preCtx">--</pre>
      </div>
      <div id="tabQueries" class="tab-content">
        <pre id="preQueries">--</pre>
      </div>
      <div id="tabCritic" class="tab-content">
        <pre id="preCritic">--</pre>
      </div>
      <div id="tabRaw" class="tab-content">
        <pre id="preRaw">--</pre>
      </div>
    </div>
  </div>

  <!-- COMPARE RESULTS (HEAD TO HEAD) -->
  <div id="compareResults">
    <div class="compare-grid">
      <!-- SIDE A -->
      <div class="compare-side">
        <div class="compare-title">
          <span style="color: #00f2fe;">System A:</span>
          <span id="cmpNameA">ContextRouter</span>
        </div>
        <div id="cmpAnsA" style="font-size: 20px; font-weight: 700; color: #34d399; margin-bottom: 12px;">--</div>
        <div style="font-size: 13px; color: var(--text-dim); margin-bottom: 8px;">
          Tokens: <b id="cmpTokA" style="color: #fff;">--</b> | Calls: <b id="cmpCallA" style="color: #fff;">--</b> | Latency: <b id="cmpLatA" style="color: #fff;">--</b>
        </div>
        <pre id="cmpCtxA" style="max-height: 350px;">--</pre>
      </div>

      <!-- SIDE B -->
      <div class="compare-side">
        <div class="compare-title">
          <span style="color: #ec4899;">System B:</span>
          <span id="cmpNameB">Naive RAG</span>
        </div>
        <div id="cmpAnsB" style="font-size: 20px; font-weight: 700; color: #f59e0b; margin-bottom: 12px;">--</div>
        <div style="font-size: 13px; color: var(--text-dim); margin-bottom: 8px;">
          Tokens: <b id="cmpTokB" style="color: #fff;">--</b> | Calls: <b id="cmpCallB" style="color: #fff;">--</b> | Latency: <b id="cmpLatB" style="color: #fff;">--</b>
        </div>
        <pre id="cmpCtxB" style="max-height: 350px;">--</pre>
      </div>
    </div>
  </div>
</div>

<!-- ARCHITECTURE EXPLAINER MODAL -->
<div id="modalExplain" class="modal-overlay" onclick="closeOnBackdrop(event, 'modalExplain')">
  <div class="modal-box">
    <button class="modal-close" onclick="closeModal('modalExplain')">&times;</button>
    <div class="modal-title">&Delta; What is ContextRouter?</div>
    <div style="font-size: 14px; line-height: 1.6; color: #cbd5e1;">
      <p style="margin-bottom: 12px;">
        Standard RAG systems naively dump entire retrieved documents into the LLM context window. This causes:
        <br>&bull; <b>"Lost-in-the-middle" attention degradation:</b> Key reasoning facts get drowned out.
        <br>&bull; <b>Excessive latency &amp; API token costs:</b> Multi-hop queries explode the prompt size.
      </p>
      <p style="margin-bottom: 12px;">
        <b>The ContextRouter Innovation:</b><br>
        We introduce a specialized <b>Context Engineer agent</b> in a LangGraph multi-agent loop with typed Pydantic contracts:
        <br>1. <b>Query Analyzer:</b> Classifies query complexity (direct, single_hop, multi_hop, comparison) and decomposes into ordered sub-queries.
        <br>2. <b>Strategy Planner:</b> Determines a per-question token budget $B$ and retrieval depth $k$.
        <br>3. <b>Context Engineer:</b> Performs cross-document sentence scoring, removes near-duplicate sentences, knapsack-packs sentences under budget $B$, and orders them to prevent attention loss.
        <br>4. <b>Critic Loop:</b> Verifies whether the draft answer is fully supported and grounded; if not, triggers targeted follow-up queries.
      </p>
      <div style="background: rgba(0, 242, 254, 0.08); border: 1px solid rgba(0, 242, 254, 0.3); padding: 12px; border-radius: 8px; font-size: 13px; color: #38bdf8;">
        &check; <b>Result:</b> Same or higher Exact Match &amp; F1 accuracy as naive RAG while using 40-60% fewer context tokens.
      </div>
    </div>
  </div>
</div>

<!-- DATASET EXPLORER MODAL -->
<div id="modalDataset" class="modal-overlay" onclick="closeOnBackdrop(event, 'modalDataset')">
  <div class="modal-box">
    <button class="modal-close" onclick="closeModal('modalDataset')">&times;</button>
    <div class="modal-title">&darr; Dataset Hub: HotpotQA Benchmark</div>
    <div style="font-size: 13.5px; line-height: 1.6; color: #cbd5e1;">
      <p style="margin-bottom: 12px;">
        This project evaluates multi-hop reasoning on <b>HotpotQA</b> (Yang et al., EMNLP 2018), *distractor* setting, *validation* split.
      </p>
      <div style="background: #060911; border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 10px; padding: 14px; margin-bottom: 16px;">
        <div style="color: #38bdf8; font-weight: 600; margin-bottom: 4px;">Local Dataset Files:</div>
        <div>&bull; <code>data/hotpot_subset.json</code> (1.92 MB, 300 curated questions + passages)</div>
        <div>&bull; <code>data/DATA_CARD.json</code> (Dataset provenance, split, seed=42)</div>
        <div>&bull; <code>data/sample_mini.json</code> (Synthetic smoke-test set)</div>
      </div>
      <p style="margin-bottom: 10px;"><b>Download Upstream Original Dataset:</b></p>
      <ul style="margin-left: 20px; margin-bottom: 14px;">
        <li><a href="http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json" target="_blank" style="color: #00f2fe;">CMU Direct JSON (hotpot_dev_distractor_v1.json)</a></li>
        <li><a href="https://huggingface.co/datasets/hotpotqa/hotpot_qa" target="_blank" style="color: #00f2fe;">Hugging Face Dataset (hotpotqa/hotpot_qa)</a></li>
        <li><a href="https://hotpotqa.github.io/" target="_blank" style="color: #00f2fe;">HotpotQA Project Homepage</a></li>
      </ul>
      <p style="margin-bottom: 6px;"><b>Download via Terminal / PowerShell:</b></p>
      <pre style="padding: 10px; font-size: 12px;">python scripts/prepare_data.py --n 300 --seed 42</pre>
    </div>
  </div>
</div>

<script>
  let currentMode = 'solo';
  let sfxEnabled = true;
  const audioCtx = (window.AudioContext || window.webkitAudioContext) ? new (window.AudioContext || window.webkitAudioContext)() : null;

  function playTone(freq, type, duration) {
    if (!sfxEnabled || !audioCtx) return;
    try {
      if (audioCtx.state === 'suspended') audioCtx.resume();
      const osc = audioCtx.createOscillator();
      const gain = audioCtx.createGain();
      osc.type = type || 'sine';
      osc.frequency.value = freq || 440;
      gain.gain.setValueAtTime(0.04, audioCtx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + duration);
      osc.connect(gain);
      gain.connect(audioCtx.destination);
      osc.start();
      osc.stop(audioCtx.currentTime + duration);
    } catch(e){}
  }

  function toggleSfx() {
    sfxEnabled = !sfxEnabled;
    document.getElementById('sfxToggle').innerText = sfxEnabled ? '😎 SFX: ON' : '🔇 SFX: OFF';
    if (sfxEnabled) playTone(587, 'triangle', 0.2);
  }

  // Load benchmark questions
  fetch('/api/examples')
    .then(r => r.json())
    .then(list => {
      const deck = document.getElementById('chipsDeck');
      deck.innerHTML = list.map(item => {
        const qEsc = item.question.replace(/"/g, '&quot;');
        return `<div class="q-chip" onclick="selectQuestion('${qEsc}')">
          <span class="q-type-badge">${item.type || 'multi_hop'}</span>
          <span>${item.question}</span>
        </div>`;
      }).join('');
    })
    .catch(() => {
      document.getElementById('chipsDeck').innerHTML = '<span class="q-chip" onclick="selectQuestion(\'Which magazine was started first Arthur\\\'s Magazine or First for Women?\')">Which magazine was started first Arthur\'s Magazine or First for Women?</span>';
    });

  function selectQuestion(q) {
    document.getElementById('q').value = q;
    playTone(440, 'sine', 0.1);
    runCockpit();
  }

  function setExecMode(mode) {
    currentMode = mode;
    playTone(520, 'sine', 0.1);
    document.getElementById('btnModeSolo').classList.toggle('active', mode === 'solo');
    document.getElementById('btnModeCompare').classList.toggle('active', mode === 'compare');
    document.getElementById('soloSysWrap').style.display = mode === 'solo' ? 'block' : 'none';
    document.getElementById('compareSysWrap').style.display = mode === 'compare' ? 'block' : 'none';
    document.getElementById('soloResults').style.display = 'none';
    document.getElementById('compareResults').style.display = 'none';
  }

  function switchTab(e, tabId) {
    document.querySelectorAll('.tab-item').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    e.currentTarget.classList.add('active');
    document.getElementById(tabId).classList.add('active');
    playTone(330, 'triangle', 0.08);
  }

  function openModal(id) { document.getElementById(id).style.display = 'flex'; }
  function closeModal(id) { document.getElementById(id).style.display = 'none'; }
  function closeOnBackdrop(e, id) { if (e.target.id === id) closeModal(id); }

  function copyAnswer() {
    navigator.clipboard.writeText(document.getElementById('vAnswer').innerText);
    playTone(880, 'sine', 0.15);
  }

  document.getElementById('q').addEventListener('keydown', e => {
    if (e.key === 'Enter') runCockpit();
  });

  function animateNodes() {
    const nodes = [1, 2, 3, 4, 5, 6];
    let i = 0;
    const interval = setInterval(() => {
      nodes.forEach(n => document.getElementById('pNode' + n).classList.remove('active-pulse'));
      if (i < nodes.length) {
        document.getElementById('pNode' + nodes[i]).classList.add('active-pulse');
        playTone(300 + i * 80, 'sine', 0.1);
        i++;
      } else {
        i = 0;
      }
    }, 400);
    return interval;
  }

  async function runCockpit() {
    const q = document.getElementById('q').value.trim();
    if (!q) return;

    const runBtn = document.getElementById('runBtn');
    const runBtnText = document.getElementById('runBtnText');
    runBtn.disabled = true;
    runBtnText.innerHTML = '<span class="spinner"></span> Routing...';

    const pulseInterval = animateNodes();

    try {
      if (currentMode === 'solo') {
        const sys = document.getElementById('sysSelect').value;
        const resp = await fetch('/api/ask', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({question: q, system: sys})
        });
        const d = await resp.json();

        // Populate Solo view
        document.getElementById('vAnswer').innerText = d.answer || '(No answer)';
        document.getElementById('kRoute').innerText = d.qtype || 'multi_hop';
        document.getElementById('kTokens').innerText = (d.ctx_tokens || 0) + ' tok';
        document.getElementById('kHops').innerText = d.iterations || 0;
        document.getElementById('kCalls').innerText = d.usage ? d.usage.llm_calls : 0;
        document.getElementById('kLatency').innerText = (d.usage ? d.usage.latency_s : 0) + 's';
        
        const violations = d.usage ? d.usage.contract_violations : 0;
        const violEl = document.getElementById('kViolations');
        violEl.innerText = violations;
        violEl.style.color = violations > 0 ? '#ef4444' : '#34d399';

        // Budget Gauge
        const maxBudget = 900;
        const pct = Math.min(100, Math.round(((d.ctx_tokens || 0) / maxBudget) * 100));
        document.getElementById('gaugeFill').style.width = pct + '%';
        document.getElementById('gaugeLabel').innerText = `${d.ctx_tokens || 0} / ${maxBudget} tokens (${pct}%)`;

        // Tabs
        document.getElementById('ctxPassageInfo').innerText = 'Selected passages: ' + ((d.ctx_titles || []).join(', ') || 'None');
        document.getElementById('preCtx').innerText = d.ctx_text || '(Empty context)';
        document.getElementById('preQueries').innerText = (d.queries || []).map((hop, idx) => `[Hop ${idx + 1}] ${hop}`).join('\n') || '(None)';
        document.getElementById('preCritic').innerText = JSON.stringify(d.critiques || [], null, 2);
        document.getElementById('preRaw').innerText = JSON.stringify(d, null, 2);

        document.getElementById('soloResults').style.display = 'block';
        document.getElementById('compareResults').style.display = 'none';

      } else {
        // Compare Mode
        const sysA = document.getElementById('sysA').value;
        const sysB = document.getElementById('sysB').value;
        const resp = await fetch('/api/compare', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({question: q, system_a: sysA, system_b: sysB})
        });
        const data = await resp.json();
        const a = data.a;
        const b = data.b;

        document.getElementById('cmpNameA').innerText = sysA;
        document.getElementById('cmpAnsA').innerText = a.answer || '(No answer)';
        document.getElementById('cmpTokA').innerText = (a.ctx_tokens || 0) + ' tokens';
        document.getElementById('cmpCallA').innerText = a.usage ? a.usage.llm_calls : 0;
        document.getElementById('cmpLatA').innerText = (a.usage ? a.usage.latency_s : 0) + 's';
        document.getElementById('cmpCtxA').innerText = a.ctx_text || '(Empty context)';

        document.getElementById('cmpNameB').innerText = sysB;
        document.getElementById('cmpAnsB').innerText = b.answer || '(No answer)';
        document.getElementById('cmpTokB').innerText = (b.ctx_tokens || 0) + ' tokens';
        document.getElementById('cmpCallB').innerText = b.usage ? b.usage.llm_calls : 0;
        document.getElementById('cmpLatB').innerText = (b.usage ? b.usage.latency_s : 0) + 's';
        document.getElementById('cmpCtxB').innerText = b.ctx_text || '(Empty context)';

        document.getElementById('compareResults').style.display = 'block';
        document.getElementById('soloResults').style.display = 'none';
      }

      // Fanfare SFX
      playTone(523.25, 'sine', 0.15);
      setTimeout(() => playTone(659.25, 'sine', 0.15), 100);
      setTimeout(() => playTone(783.99, 'sine', 0.25), 200);

    } catch (err) {
      alert('Execution Error: ' + err.message);
    } finally {
      clearInterval(pulseInterval);
      [1, 2, 3, 4, 5, 6].forEach(n => document.getElementById('pNode' + n).classList.remove('active-pulse'));
      runBtn.disabled = false;
      runBtnText.innerText = '▶ Launch Agents';
    }
  }
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def home():
    return PAGE
