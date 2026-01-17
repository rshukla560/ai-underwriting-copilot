import time
from app.core.rag.document_processor import process_pdf
from app.core.rag.embedder import embed_chunks
from app.core.rag.vector_store import upsert_chunks
from app.core.pipeline.extractor import extract_application_data
from app.core.pipeline.retriever import retrieve_context
from app.core.pipeline.risk_scorer import score_risk
from app.core.pipeline.recommender import generate_recommendation
from app.utils import get_logger

logger = get_logger(__name__)


def run_pipeline(pdf_path: str, applicant_id: str) -> dict:
    """
    Main entry point for the full underwriting pipeline.

    Orchestrates all steps in sequence — document processing, embedding,
    extraction, retrieval, risk scoring, and recommendation. Collects
    latency and cost traces from every step for observability.

    Args:
        pdf_path:     path to the uploaded PDF application
        applicant_id: unique identifier for this applicant/case

    Returns:
        dict with keys:
            - applicant_id (str): echo back for reference
            - applicant_summary (dict): extracted fields from extractor
            - risk_scores (dict): scored dimensions from risk_scorer
            - risk_level (str): low / moderate / high / uninsurable
            - recommendation (dict): decision, reasoning, red_flags, citations
            - pipeline_metrics (dict): per-step latency and total cost

    Raises:
        ValueError: if pdf_path or applicant_id is empty
        RuntimeError: if any pipeline step fails
    """
    if not pdf_path or not pdf_path.strip():
        raise ValueError("PDF path cannot be empty")

    if not applicant_id or not applicant_id.strip():
        raise ValueError("Applicant ID cannot be empty")

    pipeline_start = time.perf_counter()
    traces = {}

    logger.info(
        f"Pipeline started | "
        f"applicant_id={applicant_id} | "
        f"pdf_path={pdf_path}"
    )

    # Step 1: Document processing ──────────────────────────────────────────
    # Extract, clean, chunk PDF — no LLM call, pure text processing
    step_start = time.perf_counter()

    chunks = process_pdf(pdf_path)
    chunks = embed_chunks(chunks)
    upsert_chunks(chunks, applicant_id)

    traces["ingestion_ms"] = round((time.perf_counter() - step_start) * 1000)

    #Join all chunk texts for extraction — extractor needs full document view
    document_text = " ".join([chunk["text"] for chunk in chunks])

    logger.info(f"Step 1 complete | ingestion_ms={traces['ingestion_ms']}")

    # Step 2: Extract structured fields ────────────────────────────────────
    # LLM call #1 — distil raw text into clean structured applicant data
    step_start = time.perf_counter()

    extraction_result = extract_application_data(document_text)
    applicant_data    = extraction_result["data"]
    traces["extraction_ms"]      = round((time.perf_counter() - step_start) * 1000)
    traces["extraction_cost"]    = extraction_result["trace"]["cost_usd"]
    traces["extraction_prompt"]  = extraction_result["prompt_version"]

    logger.info(f"Step 2 complete | extraction_ms={traces['extraction_ms']}")

    # Step 3: Retrieve relevant context ────────────────────────────────────
    # No LLM call — pure vector similarity search in ChromaDB
    step_start = time.perf_counter()

    retrieval_result = retrieve_context(applicant_data, applicant_id)
    context_text     = retrieval_result["context_text"]
    traces["retrieval_ms"] = round((time.perf_counter() - step_start) * 1000)

    logger.info(f"Step 3 complete | retrieval_ms={traces['retrieval_ms']}")

    # Step 4: Score risk dimensions ────────────────────────────────────────
    # LLM call #2 — score health, financial, behavioral, occupation
    step_start = time.perf_counter()

    scoring_result       = score_risk(applicant_data, context_text)
    risk_scores          = scoring_result["scores"]
    risk_level           = scoring_result["risk_level"]
    traces["scoring_ms"]      = round((time.perf_counter() - step_start) * 1000)
    traces["scoring_cost"]    = scoring_result["trace"]["cost_usd"]
    traces["scoring_prompt"]  = scoring_result["prompt_version"]

    logger.info(f"Step 4 complete | scoring_ms={traces['scoring_ms']}")

    # ── Step 5: Generate recommendation ──────────────────────────────────────
    # LLM call #3 — final decision with reasoning, red flags, citations
    step_start = time.perf_counter()

    recommendation_result = generate_recommendation(
        applicant_data=applicant_data,
        risk_scores=risk_scores,
        context_text=context_text
    )
    recommendation  = recommendation_result["recommendation"]
    traces["recommendation_ms"]     = round((time.perf_counter() - step_start) * 1000)
    traces["recommendation_cost"]   = recommendation_result["trace"]["cost_usd"]
    traces["recommendation_prompt"] = recommendation_result["prompt_version"]

    logger.info(f"Step 5 complete | recommendation_ms={traces['recommendation_ms']}")

    # ── Collect pipeline metrics ──────────────────────────────────────────────
    total_latency_ms = round((time.perf_counter() - pipeline_start) * 1000)
    total_cost_usd   = round(
        traces.get("extraction_cost", 0) +
        traces.get("scoring_cost", 0) +
        traces.get("recommendation_cost", 0),
        6
    )

    pipeline_metrics = {
        "total_latency_ms": total_latency_ms,
        "total_cost_usd":   total_cost_usd,
        "steps": {
            "ingestion_ms":       traces["ingestion_ms"],
            "extraction_ms":      traces["extraction_ms"],
            "retrieval_ms":       traces["retrieval_ms"],
            "scoring_ms":         traces["scoring_ms"],
            "recommendation_ms":  traces["recommendation_ms"]
        },
        "prompt_versions": {
            "extraction":     traces["extraction_prompt"],
            "scoring":        traces["scoring_prompt"],
            "recommendation": traces["recommendation_prompt"]
        }
    }

    logger.info(
        f"Pipeline complete | "
        f"applicant_id={applicant_id} | "
        f"total_latency={total_latency_ms}ms | "
        f"total_cost=${total_cost_usd} | "
        f"decision={recommendation.get('decision', 'unknown')}"
    )

    return {
        "applicant_id":      applicant_id,
        "applicant_summary": applicant_data,
        "risk_scores":       risk_scores,
        "risk_level":        risk_level,
        "recommendation":    recommendation,
        "retrieved_chunks":  retrieval_result["chunks"],
        "pipeline_metrics":  pipeline_metrics
    }