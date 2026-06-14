import json
from app.core.pipeline.orchestrator import run_pipeline
from app.core.evaluation.eval_runner import run_evaluation

# Step 1 — run pipeline
print("Running pipeline...")
pipeline_result = run_pipeline(
    pdf_path="tests/fixtures/sample_application.pdf",
    applicant_id="TEST_003"
)
print(f"Pipeline complete | decision={pipeline_result['recommendation'].get('decision')}")

# Step 2 — run evaluation
print("Running evaluation...")
eval_result = run_evaluation(pipeline_result)

report  = eval_result["report"]
summary = eval_result["summary"]

print(f"Evaluation Results ")
print(f"Faithfulness:  {report.faithfulness.faithfulness_score}")
print(f"Completeness:  {report.completeness.completeness_score}")
print(f"Citations:     {report.citations.citation_accuracy}")
print(f"Action:        {summary['action']}")
print(f"Flags:         {summary['flags']}")

if report.faithfulness.unsupported_claims:
    print(f"\nUnsupported claims:")
    for claim in report.faithfulness.unsupported_claims:
        print(f" {claim}")

if report.completeness.missing_fields:
    print(f"\nMissing fields: {report.completeness.missing_fields}")

print(f"\nEval Metrics")
print(f"Faithfulness latency: {report.eval_metrics.faithfulness_latency_ms}ms")
print(f"Faithfulness cost:    ${report.eval_metrics.faithfulness_cost_usd}")