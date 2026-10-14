"""OpenAI-compatible chat client with disk cache, client-side rate limiting, retries,
a usage Meter, and an offline mock (for smoke tests only)."""
from __future__ import annotations
import hashlib, json, re, threading, time
from dataclasses import dataclass
from pathlib import Path
from pydantic import ValidationError
from .config import get_settings


@dataclass
class LLMResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency: float
    cached: bool = False


@dataclass
class Meter:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency: float = 0.0
    violations: int = 0

    def add(self, r: LLMResult):
        self.calls += 1
        self.prompt_tokens += r.prompt_tokens
        self.completion_tokens += r.completion_tokens
        self.latency += r.latency


def parse_json(text: str):
    if not text:
        return None
    t = text.strip()
    t = re.sub(r"^```(?:json)?|```$", "", t, flags=re.M).strip()
    try:
        return json.loads(t)
    except Exception:
        pass
    m = re.search(r"\{.*\}", t, flags=re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None


class LLM:
    def __init__(self, settings=None):
        self.s = settings or get_settings()
        self._lock = threading.Lock()
        self._rl = threading.Lock()
        self._last = 0.0
        self.cache: dict = {}
        self.client = None
        if self.s.mock:
            return
        if not self.s.api_key:
            raise RuntimeError("LLM_API_KEY is not set. Copy .env.example to .env and add your key "
                               "(or set MOCK_LLM=1 for an offline smoke test).")
        self.cache_path = Path(self.s.cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        if self.cache_path.exists():
            for line in self.cache_path.read_text().splitlines():
                try:
                    d = json.loads(line)
                    self.cache[d["k"]] = d
                except Exception:
                    continue
        from openai import OpenAI
        self.client = OpenAI(base_url=self.s.base_url, api_key=self.s.api_key, timeout=90, max_retries=0)

    # ------------------------------------------------------------------ real call
    def _throttle(self):
        gap = 60.0 / max(1, self.s.rpm)
        with self._rl:
            wait = self._last + gap - time.time()
            if wait > 0:
                time.sleep(wait)
            self._last = time.time()

    def chat(self, system: str, user: str, kind: str = "fast", json_mode: bool = True,
             max_tokens: int = 200) -> LLMResult:
        if self.s.mock:
            return self._mock(system, user)
        model = self.s.main_model if kind == "main" else self.s.fast_model
        key = hashlib.sha256(f"{model}|{json_mode}|{max_tokens}|{system}|{user}".encode()).hexdigest()
        hit = self.cache.get(key)
        if hit:
            return LLMResult(hit["text"], hit["pt"], hit["ct"], hit["lat"], True)
        last = None
        for attempt in range(8):
            self._throttle()
            t0 = time.time()
            try:
                kw = dict(model=model, temperature=0, max_tokens=max_tokens,
                          messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
                if json_mode:
                    kw["response_format"] = {"type": "json_object"}
                resp = self.client.chat.completions.create(**kw)
                lat = time.time() - t0
                u = resp.usage
                res = LLMResult(resp.choices[0].message.content or "",
                                getattr(u, "prompt_tokens", 0) or 0,
                                getattr(u, "completion_tokens", 0) or 0, lat)
                rec = {"k": key, "text": res.text, "pt": res.prompt_tokens, "ct": res.completion_tokens,
                       "lat": lat, "model": model}
                with self._lock:
                    self.cache[key] = rec
                    with open(self.cache_path, "a") as f:
                        f.write(json.dumps(rec) + "\n")
                return res
            except Exception as e:  # rate limit, transient, or json-mode rejection
                last = e
                msg = str(e).lower()
                if json_mode and ("json" in msg or "response_format" in msg) and "rate" not in msg:
                    json_mode = False
                    continue
                time.sleep(min(60, 2 ** attempt * 2))
        raise RuntimeError(f"LLM call failed after retries: {last}")

    # ------------------------------------------------------------------ offline mock
    def _mock(self, system: str, user: str) -> LLMResult:
        """Deterministic heuristics so the whole pipeline can be exercised offline.
        NOT a language model - never report numbers produced in mock mode."""
        s = system
        if "ROLE=ANALYZER" in s:
            q = user.split("Question:", 1)[-1].strip()
            n = len(q.split())
            if n > 14:
                out = {"qtype": "multi_hop", "sub_questions": [q]}
            elif n > 8:
                out = {"qtype": "comparison", "sub_questions": [q]}
            else:
                out = {"qtype": "single_hop", "sub_questions": []}
        elif "ROLE=ANSWERER" in s:
            ctx = user.split("CONTEXT:", 1)[-1].split("QUESTION:", 1)[0]
            q = user.split("QUESTION:", 1)[-1].lower()
            cands = [c for c in re.findall(r"\b[A-Z][a-z]+\b", ctx) if c.lower() not in q]
            out = {"answer": cands[-1] if cands else "unknown", "confidence": 0.6}
        elif "ROLE=CLOSEDBOOK" in s:
            out = {"answer": "unknown", "confidence": 0.1}
        elif "ROLE=CRITIC" in s:
            n_titles = len(re.findall(r"\[[^\]]+\]", user))
            out = {"grounded": True, "sufficient": n_titles >= 2,
                   "followup_query": "" if n_titles >= 2 else user.split("QUESTION:", 1)[-1].split("\n")[0].strip()}
        else:
            out = {"supported": True}
        text = json.dumps(out)
        return LLMResult(text, int(len((system + user).split()) * 1.3), int(len(text.split()) * 1.3), 0.01)


def call_json(llm: LLM, meter: Meter, model_cls, system: str, user: str, kind: str, max_tokens: int = 200):
    """Call the LLM and validate the reply against a Pydantic contract (one repair retry)."""
    for attempt in range(2):
        u = user if attempt == 0 else user + "\n\nReturn ONLY one valid JSON object, nothing else."
        res = llm.chat(system, u, kind=kind, json_mode=True, max_tokens=max_tokens)
        meter.add(res)
        obj = parse_json(res.text)
        if isinstance(obj, dict):
            try:
                return model_cls.model_validate(obj)
            except ValidationError:
                pass
        meter.violations += 1
    return None
