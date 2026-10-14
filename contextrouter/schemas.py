"""Typed contracts exchanged between agents (Pydantic). Every LLM-produced message is
validated against one of these; a failed validation counts as a *contract violation*."""
from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field, field_validator

QTYPES = ("direct", "single_hop", "multi_hop", "comparison")


class QueryPlan(BaseModel):
    qtype: Literal["direct", "single_hop", "multi_hop", "comparison"] = "multi_hop"
    sub_questions: list[str] = Field(default_factory=list)

    @field_validator("qtype", mode="before")
    @classmethod
    def _norm(cls, v):
        v = str(v).strip().lower().replace("-", "_").replace(" ", "_")
        return v if v in QTYPES else "multi_hop"

    @field_validator("sub_questions", mode="before")
    @classmethod
    def _subs(cls, v):
        if v is None:
            return []
        if isinstance(v, str):
            return [v]
        return [str(x) for x in v][:3]

    @property
    def needs_retrieval(self) -> bool:
        return self.qtype != "direct"


class StrategyDecision(BaseModel):
    budget: int
    k: int
    queries: list[str]


class Chunk(BaseModel):
    title: str
    sentences: list[str]
    score: float = 0.0
    query: str = ""


class EngineeredContext(BaseModel):
    text: str = ""
    titles: list[str] = Field(default_factory=list)
    tokens: int = 0
    n_input_chunks: int = 0
    n_dropped_sentences: int = 0


class AnswerDraft(BaseModel):
    answer: str = ""
    confidence: float = 0.5

    @field_validator("answer", mode="before")
    @classmethod
    def _s(cls, v):
        return "" if v is None else str(v).strip()

    @field_validator("confidence", mode="before")
    @classmethod
    def _c(cls, v):
        try:
            return float(v)
        except Exception:
            return 0.5


class Critique(BaseModel):
    grounded: bool = True
    sufficient: bool = True
    followup_query: str = ""

    @field_validator("followup_query", mode="before")
    @classmethod
    def _f(cls, v):
        return "" if v is None else str(v)


class Judgement(BaseModel):
    supported: bool = False
