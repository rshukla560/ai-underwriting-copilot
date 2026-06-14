import json
import yaml
from pathlib import Path
from app.core.llm.claude_client import call_claude
from app.core.evaluation.schemas import FaithfulnessResult
from app.config import settings
from app.utils import get_logger, strip_markdown_json

logger = get_logger(__name__)

_PROMPTS_PATH = Path(__file__).parent.parent / "prompts" / "prompts.yaml"
with open(_PROMPTS_PATH, "r") as f:
    _PROMPTS = yaml.safe_load(f)

_PROMPT_CFG = _PROMPTS["faithfulness_eval_prompt"]
PROMPT_VERSION = _PROMPT_CFG["version"]


def _extract_claims(recommendation: dict) -> list[str]:
    """
    Pulls all verifiable claims from recommendation.
    Checks citation claims and red flag descriptions —
    both are factual assertions LLM makes about the applicant.
    A fabricated red flag could cause wrongful decline decision.
    """
    claims = []

    for citation in recommendation.get("citations", []):
        claim = citation.get("claim", "")
        if claim:
            claims.append(claim)

    for flag in recommendation.get("red_flags", []):
        flag_text = flag.get("flag", "")
        if flag_text:
            claims.append(flag_text)

    return claims


def check_faithfulness(recommendation: dict,chunks: list[dict]) -> FaithfulnessResult:
    """
    LLM-as-judge hallucination detection.

    Extracts all verifiable claims from citation claims and red flags.
    Sends focused input to Claude Haiku acting as independent judge.
    Judge has no memory of generating the recommendation — unbiased assessment.
    Haiku used instead of Sonnet — binary yes/no judgment per claim
    does not need primary reasoning model, 12x cheaper.

    Args:
        recommendation: output from recommender.generate_recommendation
        chunks: retrieved source chunks from vector_store

    Returns:
        FaithfulnessResult with score and supported/unsupported claim lists
    """
    if not recommendation:
        raise ValueError("Recommendation cannot be empty")

    if not chunks:
        raise ValueError("Chunks cannot be empty")

    # extract all verifiable claims — used throughout function
    all_claims = _extract_claims(recommendation)

    if not all_claims:
        logger.warning("No verifiable claims found — recommendation is unauditable")
        return {"result": FaithfulnessResult(
                faithfulness_score=0.0,
                supported_claims=[],
                unsupported_claims=[],
                reasoning="No claims or citations found — recommendation cannot be verified/auditable"),
            "trace": {"latency_ms": 0, "cost_usd": 0.0}}

    # using all_claims directly/not full recommendation — focused input for judge,
    # reduces tokens and cost and noise to llm
    claims_text = json.dumps({"claims": all_claims}, indent=2)

    parts = []
    for c in chunks:
        parts.append(f"[Page {c['page_number']}] {c['text']}")
    context_text = "\n\n".join(parts)

    user_prompt   = _PROMPT_CFG["user"].format(
        recommendation=claims_text,
        source_chunks=context_text)
    
    system_prompt = _PROMPT_CFG["system"]

    logger.info(
        f"Faithfulness judge started | "
        f"prompt_version={PROMPT_VERSION} | "
        f"claims={len(all_claims)}")

    response = call_claude(
        prompt=user_prompt,
        system_prompt=system_prompt,
        max_tokens=1000,
        temperature=0.0,                      # deterministic judgment
        model=settings.claude_judge_model )    # Haiku — simple task, 12x cheaper

    raw_text = strip_markdown_json(response["text"])

    try:
        result = json.loads(raw_text)

    except json.JSONDecodeError as e:
        logger.error(
            f"Judge returned unparseable JSON | "
            f"raw={raw_text[:200]}"
        )
        raise ValueError(
            f"Faithfulness check failed — LLM judge did not return valid JSON: {e}"
        ) from e

    supported   = result.get("supported_claims", [])
    unsupported = result.get("unsupported_claims", [])
    reasoning   = result.get("reasoning", "")

    total = len(supported) + len(unsupported)
    score = round(len(supported) / total, 4) if total > 0 else 1.0

    logger.info(
        f"Faithfulness complete | "
        f"score={score} | "
        f"supported={len(supported)} | "
        f"unsupported={len(unsupported)} | "
        f"latency={response['trace']['latency_ms']}ms | "
        f"cost=${response['trace']['cost_usd']}"
    )
    return {"result": FaithfulnessResult(
        faithfulness_score=score,
        supported_claims=supported,
        unsupported_claims=unsupported,
        reasoning=reasoning),"trace": response["trace"] }
 