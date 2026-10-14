import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

# Strategy-planner policy: query type -> (context token budget, top-k per query)
BUDGETS = {"direct": 0, "single_hop": 400, "multi_hop": 900, "comparison": 900}
TOPK = {"direct": 0, "single_hop": 4, "multi_hop": 8, "comparison": 8}


@dataclass
class Settings:
    base_url: str
    api_key: str
    main_model: str
    fast_model: str
    mock: bool
    rpm: int
    cache_path: str
    retriever: str


def get_settings() -> Settings:
    return Settings(
        base_url=os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1"),
        api_key=os.getenv("LLM_API_KEY") or os.getenv("GROQ_API_KEY", ""),
        main_model=os.getenv("MAIN_MODEL", "llama-3.3-70b-versatile"),
        fast_model=os.getenv("FAST_MODEL", "llama-3.1-8b-instant"),
        mock=os.getenv("MOCK_LLM", "0") == "1",
        rpm=int(os.getenv("LLM_RPM", "25")),
        cache_path=os.getenv("CACHE_PATH", "results/llm_cache.jsonl"),
        retriever=os.getenv("RETRIEVER", "tfidf"),
    )
