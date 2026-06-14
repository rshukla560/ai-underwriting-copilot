import json
from app.core.pipeline.orchestrator import run_pipeline

result = run_pipeline(
    pdf_path="tests/fixtures/sample_application.pdf",
    applicant_id="TEST_002"
)

print(f"✓ Applicant:       {result['applicant_summary'].get('name')}")
print(f"✓ Risk level:      {result['risk_level']}")
print(f"✓ Decision:        {result['recommendation'].get('decision')}")
print(f"✓ Confidence:      {result['recommendation'].get('confidence')}")
print(f"✓ Total latency:   {result['pipeline_metrics']['total_latency_ms']}ms")
print(f"✓ Total cost:      ${result['pipeline_metrics']['total_cost_usd']}")

print(f"\nRisk scores:")
for dimension, data in result['risk_scores'].items():
    print(f"  {dimension}: {data.get('score')} — {data.get('reasoning')}")

print(f"\nRed flags: {len(result['recommendation'].get('red_flags', []))}")
for flag in result['recommendation'].get('red_flags', []):
    print(f"  [{flag.get('severity')}] {flag.get('flag')}")

print(f"\nPipeline metrics:")
for step, ms in result['pipeline_metrics']['steps'].items():
    print(f"  {step}: {ms}ms")