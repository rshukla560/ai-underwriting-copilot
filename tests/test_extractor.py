from app.core.rag.document_processor import process_pdf
from app.core.pipeline.extractor import extract_application_data

# Step 1: Get raw text from PDF
chunks = process_pdf("tests/fixtures/sample_application.pdf")

# Combine all chunks into one document text
document_text = " ".join([chunk["text"] for chunk in chunks])
print(f"✓ Document text length: {len(document_text)} chars")

# Step 2: Extract structured data
result = extract_application_data(document_text)

print(f"✓ Prompt version: {result['prompt_version']}")
print(f"✓ Fields extracted: {list(result['data'].keys())}")
print(f"✓ Latency: {result['trace']['latency_ms']}ms")
print(f"✓ Cost: ${result['trace']['cost_usd']}")
print(f"\nExtracted data:")
import json
print(json.dumps(result['data'], indent=2))