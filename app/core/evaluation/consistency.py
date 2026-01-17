import statistics
import time
from app.core.pipeline.risk_scorer import score_risk
from app.core.evaluation.schemas import ConsistencyResult
from app.utils import get_logger

logger = get_logger(__name__)

NUMBER_OF_RUNS = 3


def check_consistency(
    applicant_data: dict,
    context_text: str
) -> dict:
    """
    Measures pipeline stability by running risk scoring 3 times
    on identical inputs and measuring variance in overall scores.

    LLMs are non-deterministic even at temperature 0.0 —
    slight variance is expected. High variance signals the pipeline
    is unstable and needs human review.

    Returns raw variance as variance is more
    informative and transparent than normalised consistency score.

    Args:
        applicant_data: extracted fields from extractor
        context_text:   retrieved chunks from retriever

    Returns:
        dict with keys:
            - result (ConsistencyResult): variance + run details
            - total_cost_usd (float): total cost across all runs
    """
    if not applicant_data:
        raise ValueError("Applicant data cannot be empty")

    if not context_text or not context_text.strip():
        raise ValueError("Context text cannot be empty")

    logger.info(
        f"Consistency check started | "
        f"runs={NUMBER_OF_RUNS} | "
        f"applicant={applicant_data.get('name', 'unknown')}"
    )

    start_time     = time.perf_counter()
    runs           = []
    overall_scores = []
    total_cost     = 0.0

    for i in range(NUMBER_OF_RUNS):
        try:
            result        = score_risk(applicant_data, context_text)
            overall_score = result["scores"].get("overall", {}).get("score", 0)
            risk_level    = result["risk_level"]
            total_cost   += result["trace"]["cost_usd"]

            runs.append({
                "run":           i + 1,
                "overall_score": overall_score,
                "risk_level":    risk_level,
                "scores":        result["scores"]
            })
            overall_scores.append(overall_score)

            logger.info(
                f"Consistency run {i + 1}/{NUMBER_OF_RUNS} | "
                f"overall_score={overall_score} | "
                f"risk_level={risk_level}"
            )

        except Exception as e:
            logger.error(f"Consistency run {i + 1} failed: {e}")
            continue

    if not overall_scores:
        raise RuntimeError("All consistency runs failed")

    # raw variance — stdev of overall scores across runs lower is better, 0.0 = perfectly consistent
    variance = round(statistics.stdev(overall_scores) if len(overall_scores) > 1 else 0.0, 4)

    inconsistent_dimensions = _find_inconsistent_dimensions(runs)

    latency_ms = round((time.perf_counter() - start_time) * 1000)

    logger.info(
        f"Consistency check complete | "
        f"variance={variance} | "
        f"scores={overall_scores} | "
        f"inconsistent={inconsistent_dimensions} | "
        f"latency={latency_ms}ms | "
        f"cost=${round(total_cost, 6)}"
    )

    return {
        "result": ConsistencyResult(
            score_variance=variance,
            runs=runs,
            inconsistent_dimensions=inconsistent_dimensions
        ),
        "latency_ms":    latency_ms,
        "total_cost_usd": round(total_cost, 6)
    }


def _find_inconsistent_dimensions(runs: list[dict]) -> list[str]:
    """
    Identifies which risk dimensions had high variance across runs.
    Helps pinpoint which part of the scoring is unstable.
    Dimension flagged if stdev > 1.0 across runs.
    """
    if len(runs) < 2:
        return []

    dimensions   = ["health", "financial", "behavioral", "occupation", "overall"]
    inconsistent = []

    for dim in dimensions:
        dim_scores = []
        for run in runs:
            score = run["scores"].get(dim, {}).get("score")
            if score is not None:
                dim_scores.append(score)

        if len(dim_scores) > 1:
            variance = statistics.stdev(dim_scores)
            if variance > 1.0:
                inconsistent.append(dim)

    return inconsistent