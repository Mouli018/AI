"""Demo UI + API.   uvicorn app:app --reload   ->  http://127.0.0.1:8000
Set DATA_FILE to choose the corpus (default: data/hotpot_subset.json, else data/sample_mini.json)."""
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

app = FastAPI(title="ContextRouter demo")


class Ask(BaseModel):
    question: str
    system: str = "contextrouter"


@app.get("/api/examples")
def examples():
    return [{"question": q["question"], "gold": q["answer"]} for q in questions[:8]]


@app.post("/api/ask")
def ask(req: Ask):
    r = run_question(GRAPHS.get(req.system, GRAPHS["contextrouter"]), req.question)
    m = r.pop("meter")
    r["usage"] = {"llm_calls": m.calls, "prompt_tokens": m.prompt_tokens, "completion_tokens": m.completion_tokens,
                  "latency_s": round(m.latency, 2), "contract_violations": m.violations}
    return r


PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>ContextRouter - Adaptive RAG & Context-Budget Engineering</title>
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #090d16;
      --card-bg: rgba(17, 24, 39, 0.7);
      --card-border: rgba(255, 255, 255, 0.08);
      --accent: #6366f1;
      --accent-glow: rgba(99, 102, 241, 0.35);
      --accent-hover: #4f46e5;
      --cyan: #06b6d4;
      --green: #10b981;
      --amber: #f59e0b;
      --text: #f3f4f6;
      --text-muted: #9ca3af;
      --code-bg: #0b1120;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Inter', system-ui, -apple-system, sans-serif;
      background: var(--bg);
      background-image: 
        radial-gradient(at 0% 0%, rgba(99, 102, 241, 0.15) 0px, transparent 50%),
        radial-gradient(at 100% 100%, rgba(6, 182, 212, 0.1) 0px, transparent 50%);
      color: var(--text);
      min-height: 100vh;
      padding: 28px 16px 60px;
    }
    .container {
      max-width: 1040px;
      margin: 0 auto;
    }
    header {
      margin-bottom: 24px;
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      flex-wrap: wrap;
      gap: 16px;
    }
    .badge {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 4px 10px;
      border-radius: 9999px;
      font-size: 11px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      background: rgba(99, 102, 241, 0.12);
      color: #818cf8;
      border: 1px solid rgba(99, 102, 241, 0.25);
    }
    .pulse-dot {
      width: 7px;
      height: 7px;
      border-radius: 50%;
      background: #10b981;
      box-shadow: 0 0 10px #10b981;
      animation: pulse 2s infinite;
    }
    @keyframes pulse { 0%, 100% { opacity: 1; transform: scale(1); } 50% { opacity: 0.4; transform: scale(0.9); } }
    h1 {
      font-size: 28px;
      font-weight: 800;
      letter-spacing: -0.02em;
      background: linear-gradient(135deg, #ffffff 30%, #a5b4fc 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      margin-top: 6px;
    }
    .subtitle {
      color: var(--text-muted);
      font-size: 14px;
      margin-top: 4px;
    }
    .pipeline-bar {
      display: flex;
      align-items: center;
      gap: 8px;
      overflow-x: auto;
      padding: 10px 14px;
      background: rgba(17, 24, 39, 0.5);
      backdrop-filter: blur(8px);
      border: 1px solid var(--card-border);
      border-radius: 12px;
      margin-bottom: 20px;
      font-size: 12px;
      color: var(--text-muted);
    }
    .pipe-step {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      white-space: nowrap;
      font-weight: 500;
      color: #cbd5e1;
    }
    .pipe-step.highlight {
      color: #38bdf8;
      font-weight: 600;
    }
    .pipe-arrow { color: #475569; }
    .card {
      background: var(--card-bg);
      backdrop-filter: blur(12px);
      border: 1px solid var(--card-border);
      border-radius: 16px;
      padding: 20px;
      margin-bottom: 20px;
      box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
    }
    .input-group {
      display: flex;
      flex-direction: column;
      gap: 12px;
    }
    .controls-row {
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
    }
    select, input, button {
      font-family: inherit;
      font-size: 14px;
    }
    select {
      background: #111827;
      color: #f3f4f6;
      border: 1px solid rgba(255, 255, 255, 0.12);
      padding: 10px 14px;
      border-radius: 10px;
      outline: none;
      cursor: pointer;
      min-width: 220px;
    }
    select:focus { border-color: var(--accent); }
    .search-box {
      display: flex;
      gap: 10px;
      position: relative;
    }
    input#q {
      flex: 1;
      background: #0f172a;
      border: 1px solid rgba(255, 255, 255, 0.12);
      border-radius: 12px;
      padding: 14px 18px;
      color: #fff;
      font-size: 15px;
      outline: none;
      transition: all 0.2s;
    }
    input#q:focus {
      border-color: var(--accent);
      box-shadow: 0 0 0 3px var(--accent-glow);
    }
    button.btn-primary {
      background: linear-gradient(135deg, #6366f1, #4f46e5);
      color: #fff;
      border: 0;
      border-radius: 12px;
      padding: 0 26px;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s;
      display: inline-flex;
      align-items: center;
      gap: 8px;
      box-shadow: 0 4px 16px var(--accent-glow);
    }
    button.btn-primary:hover:not(:disabled) {
      transform: translateY(-1px);
      box-shadow: 0 6px 20px var(--accent-glow);
    }
    button.btn-primary:disabled {
      opacity: 0.6;
      cursor: not-allowed;
    }
    .examples-wrap {
      margin-top: 14px;
    }
    .examples-label {
      font-size: 12px;
      font-weight: 600;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.05em;
      margin-bottom: 8px;
      display: block;
    }
    .chips {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }
    .chip {
      background: rgba(255, 255, 255, 0.04);
      border: 1px solid rgba(255, 255, 255, 0.08);
      border-radius: 20px;
      padding: 6px 12px;
      font-size: 12.5px;
      color: #cbd5e1;
      cursor: pointer;
      transition: all 0.18s;
      white-space: nowrap;
      max-width: 100%;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .chip:hover {
      background: rgba(99, 102, 241, 0.16);
      border-color: rgba(99, 102, 241, 0.4);
      color: #fff;
    }
    #results { display: none; }
    .answer-card {
      background: linear-gradient(135deg, rgba(30, 41, 59, 0.8), rgba(15, 23, 42, 0.9));
      border: 1px solid rgba(99, 102, 241, 0.3);
      border-radius: 14px;
      padding: 22px;
      margin-bottom: 20px;
    }
    .answer-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 10px;
    }
    .answer-title {
      font-size: 12px;
      text-transform: uppercase;
      font-weight: 700;
      letter-spacing: 0.08em;
      color: #a5b4fc;
    }
    .ans-val {
      font-size: 24px;
      font-weight: 700;
      color: #34d399;
      line-height: 1.35;
    }
    .metrics-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
      gap: 12px;
      margin-bottom: 20px;
    }
    .metric-card {
      background: rgba(15, 23, 42, 0.6);
      border: 1px solid var(--card-border);
      border-radius: 12px;
      padding: 12px 14px;
    }
    .metric-label {
      font-size: 11px;
      color: var(--text-muted);
      text-transform: uppercase;
      font-weight: 600;
      letter-spacing: 0.05em;
      margin-bottom: 4px;
    }
    .metric-val {
      font-size: 18px;
      font-weight: 700;
      color: #fff;
    }
    .tag-qtype {
      display: inline-block;
      padding: 2px 8px;
      border-radius: 6px;
      background: rgba(56, 189, 248, 0.15);
      color: #38bdf8;
      border: 1px solid rgba(56, 189, 248, 0.3);
      font-size: 12px;
    }
    .tabs-nav {
      display: flex;
      gap: 4px;
      border-bottom: 1px solid rgba(255, 255, 255, 0.08);
      margin-bottom: 14px;
    }
    .tab-btn {
      background: none;
      border: none;
      color: var(--text-muted);
      padding: 8px 14px;
      font-size: 13px;
      font-weight: 600;
      cursor: pointer;
      border-bottom: 2px solid transparent;
      transition: all 0.2s;
    }
    .tab-btn.active {
      color: #fff;
      border-bottom-color: var(--accent);
    }
    .tab-pane { display: none; }
    .tab-pane.active { display: block; }
    pre {
      font-family: 'JetBrains Mono', monospace;
      font-size: 12.5px;
      background: var(--code-bg);
      border: 1px solid rgba(255, 255, 255, 0.06);
      border-radius: 10px;
      padding: 14px;
      color: #e2e8f0;
      white-space: pre-wrap;
      word-break: break-word;
      max-height: 440px;
      overflow-y: auto;
      line-height: 1.5;
    }
    .loading-spin {
      display: inline-block;
      width: 18px;
      height: 18px;
      border: 2px solid rgba(255, 255, 255, 0.2);
      border-radius: 50%;
      border-top-color: #fff;
      animation: spin 0.8s linear infinite;
    }
    @keyframes spin { to { transform: rotate(360deg); } }
    .status-msg {
      display: flex;
      align-items: center;
      gap: 12px;
      padding: 18px;
      background: rgba(99, 102, 241, 0.08);
      border: 1px solid rgba(99, 102, 241, 0.2);
      border-radius: 12px;
      color: #c7d2fe;
      font-size: 14px;
      margin-bottom: 20px;
    }
  </style>
</head>
<body>
<div class="container">
  <header>
    <div>
      <div class="badge"><span class="pulse-dot"></span> Multi-Agent RAG Orchestration</div>
      <h1>ContextRouter</h1>
      <p class="subtitle">Adaptive RAG with Agent-Level Context-Budget Engineering (LangGraph &amp; Pydantic)</p>
    </div>
    <div style="text-align: right;">
      <div class="badge" style="background: rgba(16, 185, 129, 0.1); color: #34d399; border-color: rgba(16, 185, 129, 0.25);">
        CIT Coimbatore &bull; 22MDCEL11
      </div>
      <div style="font-size: 12px; color: var(--text-muted); margin-top: 6px;">
        Evaluation Benchmark: HotpotQA
      </div>
    </div>
  </header>

  <div class="pipeline-bar">
    <span class="pipe-step">Query Analyzer</span>
    <span class="pipe-arrow">&rarr;</span>
    <span class="pipe-step">Strategy Planner</span>
    <span class="pipe-arrow">&rarr;</span>
    <span class="pipe-step">Retriever</span>
    <span class="pipe-arrow">&rarr;</span>
    <span class="pipe-step highlight">&star; Context Engineer</span>
    <span class="pipe-arrow">&rarr;</span>
    <span class="pipe-step">Answerer</span>
    <span class="pipe-arrow">&rarr;</span>
    <span class="pipe-step">Critic Loop</span>
  </div>

  <div class="card">
    <div class="input-group">
      <div class="controls-row">
        <div>
          <label style="font-size: 11px; font-weight: 600; text-transform: uppercase; color: var(--text-muted); display: block; margin-bottom: 4px;">Pipeline System</label>
          <select id="s">
            <option value="contextrouter">ContextRouter (Full: Adaptive Budget + Context Engineer)</option>
            <option value="adaptive_rag">Adaptive RAG (Routing without Context Engineer)</option>
            <option value="iter_rag">Iterative RAG (Fixed Top-K + Critic Loop)</option>
            <option value="naive_rag">Naive RAG (Fixed Top-5 Single Pass)</option>
            <option value="closed_book">Closed Book (LLM Parametric Only)</option>
            <option value="cr_fixed_budget">Ablation: Fixed Budget (900 Tokens)</option>
            <option value="cr_no_critic">Ablation: No Critic Loop</option>
          </select>
        </div>
      </div>

      <div class="search-box">
        <input id="q" type="text" placeholder="Ask a multi-hop reasoning question or select an example below..." autocomplete="off">
        <button id="askBtn" class="btn-primary" onclick="go()">
          <span id="btnText">Ask Router</span>
        </button>
      </div>

      <div class="examples-wrap">
        <span class="examples-label">Example Benchmark Questions (Click to test):</span>
        <div id="ex" class="chips">
          <span class="chip" style="opacity: 0.5;">Loading sample questions...</span>
        </div>
      </div>
    </div>
  </div>

  <div id="loader" style="display: none;" class="status-msg">
    <div class="loading-spin"></div>
    <div>
      <strong>Orchestrating agents...</strong>
      <div style="font-size: 12px; color: var(--text-muted); margin-top: 2px;">
        Executing Query Analyzer &rarr; Context Engineering &rarr; Critic Loop via Groq LLaMA 3.3
      </div>
    </div>
  </div>

  <div id="results">
    <div class="answer-card">
      <div class="answer-header">
        <span class="answer-title">Synthesized Answer</span>
        <button onclick="navigator.clipboard.writeText(document.getElementById('ansText').innerText)" style="background: rgba(255,255,255,0.08); border: 1px solid var(--card-border); color: #fff; padding: 4px 10px; border-radius: 6px; cursor: pointer; font-size: 11px;">Copy Answer</button>
      </div>
      <div id="ansText" class="ans-val">--</div>
    </div>

    <div class="metrics-grid">
      <div class="metric-card">
        <div class="metric-label">Route Classification</div>
        <div class="metric-val"><span id="mRoute" class="tag-qtype">--</span></div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Context Tokens</div>
        <div id="mTokens" class="metric-val">--</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Critic Iterations</div>
        <div id="mIters" class="metric-val">--</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">LLM Calls</div>
        <div id="mCalls" class="metric-val">--</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Latency</div>
        <div id="mLatency" class="metric-val">--</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Schema Violations</div>
        <div id="mViolations" class="metric-val" style="color: #34d399;">0</div>
      </div>
    </div>

    <div class="card">
      <div class="tabs-nav">
        <button class="tab-btn active" onclick="setTab(event, 'tab-ctx')">Engineered Context</button>
        <button class="tab-btn" onclick="setTab(event, 'tab-queries')">Search Queries</button>
        <button class="tab-btn" onclick="setTab(event, 'tab-critique')">Critic Trace</button>
        <button class="tab-btn" onclick="setTab(event, 'tab-raw')">Full JSON Trace</button>
      </div>

      <div id="tab-ctx" class="tab-pane active">
        <div style="font-size: 12px; color: var(--text-muted); margin-bottom: 8px;" id="ctxMeta"></div>
        <pre id="ctxPre">--</pre>
      </div>

      <div id="tab-queries" class="tab-pane">
        <pre id="queriesPre">--</pre>
      </div>

      <div id="tab-critique" class="tab-pane">
        <pre id="critiquePre">--</pre>
      </div>

      <div id="tab-raw" class="tab-pane">
        <pre id="rawPre">--</pre>
      </div>
    </div>
  </div>
</div>

<script>
  // Load benchmark sample questions
  fetch('/api/examples')
    .then(r => r.json())
    .then(list => {
      const exDiv = document.getElementById('ex');
      exDiv.innerHTML = list.map(x => {
        const qEsc = x.question.replace(/"/g, '&quot;');
        return `<span class="chip" title="Gold: ${x.gold}" onclick="selectQuestion('${qEsc}')">${x.question}</span>`;
      }).join('');
    })
    .catch(() => {
      document.getElementById('ex').innerHTML = '<span class="chip" onclick="selectQuestion(\\'Which magazine was started first Arthur\\'s Magazine or First for Women?\\')">Which magazine was started first Arthur\\'s Magazine or First for Women?</span>';
    });

  function selectQuestion(text) {
    document.getElementById('q').value = text;
    go();
  }

  function setTab(e, tabId) {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
    e.currentTarget.classList.add('active');
    document.getElementById(tabId).classList.add('active');
  }

  document.getElementById('q').addEventListener('keydown', e => {
    if (e.key === 'Enter') go();
  });

  async function go() {
    const q = document.getElementById('q').value.trim();
    const system = document.getElementById('s').value;
    if (!q) return;

    const askBtn = document.getElementById('askBtn');
    const btnText = document.getElementById('btnText');
    const loader = document.getElementById('loader');
    const results = document.getElementById('results');

    askBtn.disabled = true;
    btnText.innerHTML = '<span class="loading-spin"></span> Processing...';
    loader.style.display = 'flex';
    results.style.display = 'none';

    try {
      const resp = await fetch('/api/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: q, system: system })
      });
      const d = await resp.json();

      document.getElementById('ansText').innerText = d.answer || '(No answer returned)';
      document.getElementById('mRoute').innerText = d.qtype || 'unknown';
      document.getElementById('mTokens').innerText = (d.ctx_tokens || 0) + ' tok';
      document.getElementById('mIters').innerText = d.iterations || 0;
      document.getElementById('mCalls').innerText = d.usage ? d.usage.llm_calls : 0;
      document.getElementById('mLatency').innerText = (d.usage ? d.usage.latency_s : 0) + 's';
      
      const violations = d.usage ? d.usage.contract_violations : 0;
      const violEl = document.getElementById('mViolations');
      violEl.innerText = violations;
      violEl.style.color = violations > 0 ? '#ef4444' : '#34d399';

      document.getElementById('ctxMeta').innerText = 'Selected passages: ' + ((d.ctx_titles || []).join(', ') || 'None');
      document.getElementById('ctxPre').innerText = d.ctx_text || '(Empty context)';
      document.getElementById('queriesPre').innerText = (d.queries || []).map((query, i) => `[Hop ${i+1}] ${query}`).join('\\n') || '(No queries executed)';
      document.getElementById('critiquePre').innerText = JSON.stringify(d.critiques || [], null, 2);
      document.getElementById('rawPre').innerText = JSON.stringify(d, null, 2);

      results.style.display = 'block';
    } catch (err) {
      alert('Error querying ContextRouter: ' + err.message);
    } finally {
      askBtn.disabled = false;
      btnText.innerText = 'Ask Router';
      loader.style.display = 'none';
    }
  }
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def home():
    return PAGE
