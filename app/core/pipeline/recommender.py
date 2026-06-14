import json
import yaml
from pathlib import Path
from app.core.llm.claude_client import call_claude
from app.utils import get_logger, strip_markdown_json

logger = get_logger(__name__)

# Load prompts once at module load
_PROMPTS_PATH = Path(__file__).parent.parent / "prompts" / "prompts.yaml"
with open(_PROMPTS_PATH, "r") as f:
    _PROMPTS = yaml.safe_load(f)

_PROMPT_CFG = _PROMPTS["recommendation_prompt"]
PROMPT_VERSION = _PROMPT_CFG["version"]

# Valid decisions — centralised so validation is consistent across pipeline and eval
#VALID_DECISIONS = {"Approve", "Decline", "Review"} 
VALID_DECISIONS = {"approve", "decline", "review"}


def generate_recommendation(
    applicant_data: dict,
    risk_scores: dict,
    context_text: str
) -> dict:
    """
    LLM call #3 in the pipeline — generates final underwriting recommendation.

    Takes extracted applicant fields, risk scores from risk_scorer, and
    retrieved document context. Returns a structured decision with full
    reasoning, red flags with severity, and citations linking every claim
    to source documents.

    Args:
        applicant_data: extracted fields from extractor.extract_application_data
        risk_scores:    scored dimensions from risk_scorer.score_risk
        context_text:   joined chunk texts from retriever.retrieve_context

    Returns:
        dict with keys:
            - recommendation (dict): decision, confidence, premium_range,
                                     reasoning, red_flags, citations
            - prompt_version (str): prompt version for traceability
            - trace (dict): latency, tokens, cost from claude_client

    Raises:
        ValueError: if inputs empty, LLM returns unparseable JSON,
                    or decision is not one of Approve/Decline/Review
        RuntimeError: if Claude API call fails
    """
    if not applicant_data:
        raise ValueError("Applicant data cannot be empty")

    if not risk_scores:
        raise ValueError("Risk scores cannot be empty")

    #validating risk_score if empty
    overall_score = risk_scores.get("overall", {}).get("score")
    if overall_score is not None and not 0 <= overall_score <= 10:
        raise ValueError(
            f"Invalid overall score {overall_score} — must be between 0 and 10")

    if not context_text or not context_text.strip():
        raise ValueError("Context text cannot be empty")

    # Serialise both dicts to formatted JSON strings for prompt injection
    applicant_json = json.dumps(applicant_data, indent=2)
    scores_json    = json.dumps(risk_scores, indent=2)

    user_prompt = _PROMPT_CFG["user"].format(
        applicant_data=applicant_json,
        risk_scores=scores_json,
        retrieved_context=context_text
    )
    system_prompt = _PROMPT_CFG["system"]

    logger.info(
        f"Generating recommendation | "
        f"prompt_version={PROMPT_VERSION} | "
        f"applicant={applicant_data.get('name', 'unknown')} | "
        f"overall_risk={risk_scores.get('overall', {}).get('score', 'unknown')}"
    )

    response = call_claude(
        prompt=user_prompt,
        system_prompt=system_prompt,
        max_tokens=1500,   # recommendations are longer — citations and reasoning need space
        temperature=0.1    # slight variance for reasoning quality
    )

    raw_text = strip_markdown_json(response["text"])

    try:
        recommendation = json.loads(raw_text)

    except json.JSONDecodeError as e:
        logger.error(
            f"LLM returned unparseable JSON | "
            f"prompt_version={PROMPT_VERSION} | "
            f"raw_response={raw_text[:200]}"
        )
        raise ValueError(
            f"Recommendation failed — LLM did not return valid JSON: {e}"
        ) from e

    # Validate decision is one of the three allowed values
    decision_raw = recommendation.get("decision", "")
    decision = decision_raw.lower().strip()

    if decision not in VALID_DECISIONS:
        raise ValueError(
            f"LLM returned invalid decision '{decision}' — "
            f"must be one of {VALID_DECISIONS}"
        )
    # overwrite with clean value
    recommendation["decision"] = decision   

    logger.info(
        f"Recommendation complete | "
        f"prompt_version={PROMPT_VERSION} | "
        f"decision={decision} | "
        f"confidence={recommendation.get('confidence', 'unknown')} | "
        f"red_flags={len(recommendation.get('red_flags', []))}"
    )

    return {
        "recommendation": recommendation,
        "prompt_version": PROMPT_VERSION,
        "trace": response["trace"]
    }