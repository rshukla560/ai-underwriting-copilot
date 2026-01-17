import os
import shutil
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from app.core.pipeline.orchestrator import run_pipeline
from app.core.evaluation.eval_runner import run_evaluation
from app.utils import get_logger

logger = get_logger(__name__)

router = APIRouter()

# temp directory for uploaded PDFs deleted after processing — no applicant data stored on disk
UPLOAD_DIR = "/tmp/underwriting_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)



@router.post("/analyze/{applicant_id}")
async def analyze(applicant_id: str, file: UploadFile = File(...),run_consistency: bool = False):
    """
    Main endpoint — accepts PDF upload, runs full pipeline and evaluation.
    Returns structured JSON with decision, risk scores and eval metrics.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files accepted")

    if not applicant_id or not applicant_id.strip():
        raise HTTPException(status_code=400, detail="Applicant ID cannot be empty")

    pdf_path = f"{UPLOAD_DIR}/{applicant_id}.pdf"

    try:
        # save uploaded PDF to temp location
        with open(pdf_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        logger.info(
            f"PDF received | "
            f"applicant_id={applicant_id} | "
            f"filename={file.filename}"
        )

        # run full 5-step pipeline
        pipeline_result = run_pipeline(pdf_path=pdf_path, applicant_id=applicant_id)

        # run evaluation framework
        eval_result = run_evaluation(pipeline_result, run_consistency=run_consistency)

        report  = eval_result["report"]
        summary = eval_result["summary"]

        return JSONResponse(content={
            "applicant_id":  applicant_id,
            "decision":      pipeline_result["recommendation"].get("decision"),
            "confidence":    pipeline_result["recommendation"].get("confidence"),
            "risk_level":    pipeline_result["risk_level"],
            "risk_scores":   pipeline_result["risk_scores"],
            "red_flags":     pipeline_result["recommendation"].get("red_flags", []),
            "citations":     pipeline_result["recommendation"].get("citations", []),
            "reasoning":     pipeline_result["recommendation"].get("reasoning"),
            "premium_range": pipeline_result["recommendation"].get("premium_range"),
            "evaluation": {
                "faithfulness":       report.faithfulness.faithfulness_score,
                "completeness":       report.completeness.completeness_score,
                "citation_accuracy":  report.citations.citation_accuracy,
                "action":             summary["action"],
                "flags":              summary["flags"],
                "unsupported_claims": report.faithfulness.unsupported_claims
            },
            "pipeline_metrics": pipeline_result["pipeline_metrics"]
        })

    except HTTPException:
        raise

    except Exception as e:
        logger.error(
            f"Pipeline failed | "
            f"applicant_id={applicant_id} | "
            f"error={e}"
        )
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline failed: {str(e)}"
        )

    finally:
        # always clean up — no applicant data left on disk
        if os.path.exists(pdf_path):
            os.remove(pdf_path)
            logger.info(f"Temp file cleaned | {pdf_path}")