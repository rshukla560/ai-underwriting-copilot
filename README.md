# AI Underwriting Copilot
An agentic RAG pipeline for insurance underwriting risk assessment.

## Live Demo
- **Visual Demo (Hugging Face Spaces):** https://huggingface.co/spaces/rshukla560/underwriting-copilot
> Download the sample application to upload for Huggingface demo https://github.com/rshukla560/ai-underwriting-copilot/raw/master/tests/fixtures/sample_application.pdf

- **API Docs (interactive):** https://ai-underwriting-copilot-production.up.railway.app/docs

- **API Health Check:** https://ai-underwriting-copilot-production.up.railway.app/health

## How It Works
A 5-step agentic pipeline, each step using specific tools for the job:

1. **Document ingestion** — PyMuPDF extracts and chunks the PDF, OpenAI `text-embedding-3-small` generates embeddings, ChromaDB stores them as a searchable vector index

2. **Field extraction** — Claude Sonnet (Anthropic) reads raw PDF text and extracts 13 structured fields (name, medical history, occupation, income, etc.)

3. **Context retrieval** — semantic search over ChromaDB retrieves the policy chunks most relevant to the extracted fields

4. **Risk scoring** — the LLM scores health, financial, behavioral, and occupation risk dimensions using the extracted fields and retrieved context

5. **Recommendation** — the LLM generates a final decision (approve/decline/review) with citations linking every claim back to source text, plus red flags and premium range

## Evaluation Framework  runs automatically after every pipeline execution, using 4 independent metrics:

- **Faithfulness** — a second, smaller LLM (Claude Haiku) acts as an independent judge, checking each claim against source documents to detect hallucination — chosen for its lower cost on this simple binary-judgment task
- **Completeness** — rule-based check confirming all required output fields are present
- **Citation accuracy** — rule-based keyword verification that cited excerpts actually exist in source chunks
- **Consistency** *(optional)* — runs risk scoring 3x to measure score variance and pipeline stability

Backend deployed via Docker on Railway. Demo UI built with Gradio, deployed on Hugging Face Spaces.

## Observability

Every pipeline run returns full latency and cost breakdown per step as below for example : 

"pipeline_metrics": {
  "total_latency_ms": 27363,
  "total_cost_usd": 0.02694,
  "steps": {
    "ingestion_ms": 1383,
    "extraction_ms": 8021,
    "retrieval_ms": 420,
    "scoring_ms": 5783,
    "recommendation_ms": 11756
  },
  "total_cost_breakup_usd": {
    "extraction_cost": 0.005529,
    "risk_scoring_cost": 0.007152,
    "recommendation_cost": 0.014796
  }
}
```

This makes it possible to identify latency and cost bottlenecks per step, track prompt version performance over time, and reason about cost-at-scale before deploying changes.

## Project Structure
ai-underwriting-copilot/
app/          FastAPI backend, RAG pipeline, evaluation framework
frontend/     Gradio UI deployed to Hugging Face Spaces
tests/        pipeline and evaluation tests
Dockerfile    container definition

## How to Test/Postman
**Health check:**
GET https://ai-underwriting-copilot-production.up.railway.app/health

**Run the full pipeline (Postman):**

1. Open Postman, create a POST request:
https://ai-underwriting-copilot-production.up.railway.app/api/v1/analyze/{applicant_id}
2. Replace `{applicant_id}` with any ID, e.g. `TEST_001`
3. Set Body type to `form-data`
4. Add key `file` (type: File), upload an insurance application PDF
5. Click Send — response takes 20-30 seconds

A sample PDF is available at `tests/fixtures/sample_application.pdf`. Returns decision, risk scores, citations, and evaluation metrics.

## Local Setup
git clone https://github.com/rshukla560/ai-underwriting-copilot
cd ai-underwriting-copilot
cp .env.example .env
docker compose up
Add your `OPENAI_API_KEY` and `ANTHROPIC_API_KEY` to `.env` before running.

