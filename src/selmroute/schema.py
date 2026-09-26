from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class Probe(BaseModel):
    type: Literal["noul", "choice", "score"]
    instructions: str
    criteria: dict[str, str | None] | list[str] | None = None

    @field_validator("criteria")
    @classmethod
    def validate_criteria(cls, value: Any, info):
        qtype = info.data.get("type")
        if qtype in {"choice", "score"} and not value:
            raise ValueError(f"{qtype} probes require non-empty criteria")
        return value


class ProbeSet(BaseModel):
    version: str
    model: str = "jev-1.13.0"
    questions: dict[str, Probe]


class JevUsage(BaseModel):
    input_tokens: int | None = None
    output_tokens: int | None = None


class JevResponse(BaseModel):
    model: str
    answers: dict[str, dict[str, Any]]
    usage: JevUsage = Field(default_factory=JevUsage)


class DiscoveryOperation(BaseModel):
    operation: Literal["ADD", "REVISE", "DROP"]
    id: str
    question: Probe | None = None
    rationale: str

    @field_validator("question")
    @classmethod
    def require_question(cls, value: Probe | None, info):
        op = info.data.get("operation")
        if op in {"ADD", "REVISE"} and value is None:
            raise ValueError(f"{op} requires question")
        return value
