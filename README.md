# AI Underwriting Copilot

An agentic RAG pipeline for insurance underwriting risk assessment.

## Architecture

5-step pipeline:
1. Document ingestion — PDF processing, chunking, embedding, ChromaDB storage
2. Field extraction — LLM extracts 13 structured fields from raw PDF text
3. Context retrieval — semantic search retrieves relevant policy context
4. Risk scoring — LLM scores health, financial, behavioral, occupation dimensions
5. Recommendation — LLM generates decision with citations and red flags

## Evaluation Framework

4 independent metrics per case:
- Faithfulness — LLM-as-judge hallucination detection (Claude Haiku)
- Completeness — rule-based schema validation
- Citation accuracy — excerpt keyword verification
- Consistency — score variance across 3 runs (optional)

## Tech Stack

- FastAPI + uvicorn
- Claude Sonnet (pipeline) + Claude Haiku (evaluation)
- OpenAI text-embedding-3-small
- ChromaDB (vector store)
- PyMuPDF (PDF processing)
- Docker + Railway

## API Endpoints

POST /api/v1/analyze/{applicant_id}
→ Upload PDF, run full pipeline + evaluation
→ Returns decision, risk scores, citations, evaluation metrics

GET /health
→ Health check

## Local Setup

git clone https://github.com/your-username/ai-underwriting-copilot
cd backend
cp .env.example .env  # add your API keys
docker compose up

## Live Demo

API: coming soon
Demo: coming soon
