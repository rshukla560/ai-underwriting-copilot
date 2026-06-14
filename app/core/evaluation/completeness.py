from app.core.evaluation.schemas import CompletenessResult
from app.utils import get_logger

logger = get_logger(__name__)

# all fields the recommendation must contain to be considered complete
REQUIRED_FIELDS = ["decision","confidence","premium_range","reasoning","red_flags","citations"]


def check_completeness(recommendation: dict) -> CompletenessResult:
    """
    Checks all required fields are present and non-empty in the recommendation.
    Rule-based — no LLM call.
    Args:
        recommendation: output from recommender.generate_recommendation

    Returns:
        CompletenessResult with score and lists of present/missing fields
    """
    if not recommendation:
        raise ValueError("Recommendation cannot be empty")

    present_fields = []
    missing_fields = []

    for field in REQUIRED_FIELDS:
        value = recommendation.get(field)

        if value is not None and value != "" and value != []:
            present_fields.append(field)
        else:
            missing_fields.append(field)
            logger.warning(f"Missing or empty field: {field}")

    score = round(len(present_fields) / len(REQUIRED_FIELDS), 4)

    logger.info(
        f"Completeness check | "
        f"score={score} | "
        f"present={len(present_fields)} | "
        f"missing={len(missing_fields)}"
    )

    return CompletenessResult(
        completeness_score=score,
        present_fields=present_fields,
        missing_fields=missing_fields
    )