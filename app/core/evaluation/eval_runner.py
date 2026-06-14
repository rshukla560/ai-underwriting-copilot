import time
from app.core.evaluation.faithfulness import check_faithfulness
from app.core.evaluation.completeness import check_completeness
from app.core.evaluation.citation_checker import check_citations
from app.core.evaluation.consistency import check_consistency
from app.core.evaluation.schemas import EvalReport, EvalMetrics
from app.utils import get_logger

logger = get_logger(__name__)


def run_evaluation(
    pipeline_result: dict,
    run_consistency: bool = False
) -> dict:
    """
    Runs all evaluators against a completed pipeline result.
    Returns structured eval report with four independent metrics.

    Consistency check disabled by default — costs 3x scoring price.
    Enable explicitly for stability testing and prompt version audits.

    Args:
        pipeline_result: complete output from orchestrator.run_pipeline
        run_consistency: whether to run 3x consistency check

    Returns:
        dict with keys:
            - report (EvalReport): all evaluation results
            - summary (dict): human readable summary with action
    """
    if not pipeline_result:
        raise ValueError("Pipeline result cannot be empty")

    case_id        = pipeline_result.get("applicant_id", "unknown")
    recommendation = pipeline_result.get("recommendation", {})
    chunks         = pipeline_result.get("retrieved_chunks", [])
    applicant_data = pipeline_result.get("applicant_summary", {})
    context_text   = pipeline_result.get("context_text", "")

    logger.info(
        f"Evaluation started | "
        f"case_id={case_id} | "
        f"consistency={run_consistency}"
    )

    eval_start   = time.perf_counter()
    eval_metrics = EvalMetrics()

    # Completeness — rule-based, free 
    completeness_result = check_completeness(recommendation)

    #  Citation checker — rule-based, free 
    citation_result = check_citations(recommendation, chunks)

    # Faithfulness — LLM judge (Haiku) 
    faithfulness_response = check_faithfulness(recommendation, chunks)
    faithfulness_result   = faithfulness_response["result"]
    eval_metrics.faithfulness_latency_ms = faithfulness_response["trace"]["latency_ms"]
    eval_metrics.faithfulness_cost_usd   = faithfulness_response["trace"]["cost_usd"]

    # Consistency — optional, 3x Sonnet calls 
    consistency_result = None
    if run_consistency:
        consistency_response = check_consistency(applicant_data, context_text)
        consistency_result   = consistency_response["result"]
        eval_metrics.consistency_latency_ms = consistency_response["latency_ms"]
        eval_metrics.consistency_cost_usd   = consistency_response["total_cost_usd"]

    total_latency_ms = round((time.perf_counter() - eval_start) * 1000)

    report = EvalReport(
        case_id      = case_id,
        faithfulness = faithfulness_result,
        completeness = completeness_result,
        citations    = citation_result,
        consistency  = consistency_result,
        eval_metrics = eval_metrics
    )

    summary = _build_summary(report, total_latency_ms)

    logger.info(
        f"Evaluation complete | "
        f"case_id={case_id} | "
        f"faithfulness={report.faithfulness.faithfulness_score} | "
        f"completeness={report.completeness.completeness_score} | "
        f"citations={report.citations.citation_accuracy} | "
        f"action={summary['action']} | "
        f"latency={total_latency_ms}ms"
    )

    return {
        "report":  report,
        "summary": summary
    }


def _build_summary(report: EvalReport, latency_ms: int) -> dict:
    """
    Builds human readable summary of evaluation results.
    Flags dimensions that need attention.
    Recommends action based on number of flags.

    action logic:
        no flags   → auto_proceed  (high quality output)
        1 flag     → human_review  (one concern, underwriter checks)
        2+ flags   → block         (multiple concerns, full review)
    """
    flags = []

    # faithfulness below 0.80 — hallucination risk
    if report.faithfulness.faithfulness_score < 0.80:
        flags.append(
            f"low faithfulness {report.faithfulness.faithfulness_score} — "
            f"unsupported: {report.faithfulness.unsupported_claims}"
        )

    # any missing required fields
    if report.completeness.missing_fields:
        flags.append(
            f"incomplete output — missing: {report.completeness.missing_fields}"
        )

    # citation accuracy below 0.70 — fabricated evidence risk
    if report.citations.citation_accuracy < 0.70:
        flags.append(
            f"low citation accuracy {report.citations.citation_accuracy} — "
            f"invalid: {report.citations.invalid_citations}"
        )

    # high variance — unstable pipeline
    if report.consistency and report.consistency.score_variance > 1.0:
        flags.append(
            f"high variance {report.consistency.score_variance} — "
            f"unstable: {report.consistency.inconsistent_dimensions}"
        )

    if not flags:
        action = "auto_proceed"
    elif len(flags) == 1:
        action = "human_review"
    else:
        action = "block"

    return {
        "case_id":           report.case_id,
        "action":            action,
        "flags":             flags,
        "latency_ms":        latency_ms,
        "faithfulness":      report.faithfulness.faithfulness_score,
        "completeness":      report.completeness.completeness_score,
        "citation_accuracy": report.citations.citation_accuracy,
        "variance":          report.consistency.score_variance
                             if report.consistency else None
    }