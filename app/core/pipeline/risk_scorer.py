import json
import yaml
from pathlib import Path
from app.core.llm.claude_client import call_claude
from app.utils import get_logger, strip_markdown_json

logger = get_logger(__name__)

# Load prompts once at module load — same pattern as extractor.py
_PROMPTS_PATH = Path(__file__).parent.parent / "prompts" / "prompts.yaml"
with open(_PROMPTS_PATH, "r") as f:
    _PROMPTS = yaml.safe_load(f)

_PROMPT_CFG = _PROMPTS["risk_scoring_prompt"]
PROMPT_VERSION = _PROMPT_CFG["version"]

# Risk score boundaries — centralised so UI and eval use same thresholds
LOW_RISK_MAX      = 3
MODERATE_RISK_MAX = 6
HIGH_RISK_MAX     = 9
UNINSURABLE       = 10


def score_risk(applicant_data: dict, context_text: str) -> dict:
    """
    LLM call #2 in the pipeline — scores applicant risk across four dimensions.

    Args:
        applicant_data: extracted fields from extractor.extract_application_data
        context_text: joined chunk texts from retriever.retrieve_context

    Returns:
        dict with keys:
            - scores (dict): risk scores for health, financial, behavioral,
                             occupation, overall — each with score + reasoning
            - risk_level (str): low / moderate / high / uninsurable
            - prompt_version (str): prompt version for traceability
            - trace (dict): latency, tokens, cost from claude_client

    Raises:
        ValueError: if inputs are empty or LLM returns unparseable JSON
        RuntimeError: if Claude API call fails
    """
    if not applicant_data:
        raise ValueError("Applicant data cannot be empty")

    if not context_text or not context_text.strip():
        raise ValueError("Context text cannot be empty")

    # Serialise applicant data to JSON string for prompt injection
    applicant_json = json.dumps(applicant_data, indent=2)

    user_prompt = _PROMPT_CFG["user"].format(
        applicant_data=applicant_json,
        retrieved_context=context_text
    )
    system_prompt = _PROMPT_CFG["system"]

    logger.info(
        f"Scoring risk | "
        f"prompt_version={PROMPT_VERSION} | "
        f"applicant={applicant_data.get('name', 'unknown')}"
    )

    response = call_claude(
        prompt=user_prompt,
        system_prompt=system_prompt,
        max_tokens=1000,
        temperature=0.0  # deterministic — consistency eval requires stable scores
    )

    raw_text = strip_markdown_json(response["text"])

    try:
        scores = json.loads(raw_text)

    except json.JSONDecodeError as e:
        logger.error(
            f"LLM returned unparseable JSON | "
            f"prompt_version={PROMPT_VERSION} | "
            f"raw_response={raw_text[:200]}"
        )
        raise ValueError(
            f"Risk scoring failed — LLM did not return valid JSON: {e}"
        ) from e

    # Derive human-readable risk level from overall score
    overall_score = scores.get("overall", {}).get("score", 0)
    risk_level = _classify_risk_level(overall_score)

    logger.info(
        f"Risk scoring complete | "
        f"prompt_version={PROMPT_VERSION} | "
        f"overall_score={overall_score} | "
        f"risk_level={risk_level}"
    )

    return {
        "scores": scores,
        "risk_level": risk_level,
        "prompt_version": PROMPT_VERSION,
        "trace": response["trace"]
    }


def _classify_risk_level(overall_score: int) -> str:
    """
    Converts numeric overall score to human readable risk level.
    Thresholds match the scoring guide in risk_scoring_prompt.

    Args:
        overall_score: integer 0-10 from LLM risk scoring

    Returns:
        risk level string: low / moderate / high / uninsurable
    """
    if overall_score <= LOW_RISK_MAX:
        return "low"
    elif overall_score <= MODERATE_RISK_MAX:
        return "moderate"
    elif overall_score <= HIGH_RISK_MAX:
        return "high"
    else:
        return "uninsurable"