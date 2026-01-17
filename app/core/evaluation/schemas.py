from pydantic import BaseModel, Field
from typing import Optional

#defines expected shape and value ranges for faithfulness, completeness,
#citation and consistency results.
class FaithfulnessResult(BaseModel):
    faithfulness_score: float = Field(ge=0.0, le=1.0)
    supported_claims: list[str] = []
    unsupported_claims: list[str] = []
    reasoning: str


class CompletenessResult(BaseModel):
    completeness_score: float = Field(ge=0.0, le=1.0)
    present_fields: list[str] = []
    missing_fields: list[str] = []


class CitationResult(BaseModel):
    citation_accuracy: float = Field(ge=0.0, le=1.0)
    valid_citations: list[str] = []
    invalid_citations: list[str] = []


class ConsistencyResult(BaseModel):
    #consistency_score: float = Field(ge=0.0, le=1.0)
    score_variance:  float = Field(ge=0.0)  # no upper limit
    runs: list[dict] = []
    inconsistent_dimensions: list[str] = []


class EvalMetrics(BaseModel):
    faithfulness_latency_ms: int = 0
    faithfulness_cost_usd: float = 0.0
    consistency_latency_ms: int = 0
    consistency_cost_usd: float = 0.0


class EvalReport(BaseModel):
    case_id: str
    faithfulness: FaithfulnessResult
    completeness: CompletenessResult
    citations: CitationResult
    consistency: Optional[ConsistencyResult] = None
    #overall_eval_score: float = Field(ge=0.0, le=1.0)
    eval_metrics: EvalMetrics = EvalMetrics()