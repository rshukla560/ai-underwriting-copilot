from app.core.rag.embedder import embed_query
from app.core.rag.vector_store import upsert_chunks, query_similar_chunks
from app.utils import get_logger

logger = get_logger(__name__)


def retrieve_context(applicant_data: dict, applicant_id: str) -> dict:
    """
    Step 2 in the pipeline — retrieves relevant document chunks from ChromaDB
    using the extracted applicant data as the search query.

    No LLM call here — pure vector similarity search.
    Converts applicant fields into a natural language query, embeds it,
    and retrieves top-K semantically similar chunks.

    Args:
        applicant_data: extracted fields from extractor.extract_application_data
        applicant_id: used to filter ChromaDB to this applicant's docs only

    Returns:
        dict with keys:
            - chunks (list[dict]): top-K retrieved chunks with similarity scores
            - query_used (str): the query string sent to the vector store
            - context_text (str): all chunks joined as one string for prompt injection
    """
    if not applicant_data:
        raise ValueError("Applicant data cannot be empty")

    if not applicant_id or not applicant_id.strip():
        raise ValueError("Applicant ID cannot be empty")

    # Building a focused query from extracted fields — more specific than raw text
    # combines the highest-signal fields for retrieval
    medical = ", ".join(applicant_data.get("medical_history", []) or ["none"])
    occupation = applicant_data.get("occupation", "unknown")
    coverage = applicant_data.get("coverage_type", "unknown")
    age = applicant_data.get("age", "unknown")

    query = (
        f"insurance risk assessment for {coverage} applicant "
        f"age {age}, occupation {occupation}, "
        f"medical history: {medical}"
    )

    logger.info(
        f"Retrieving context | "
        f"applicant_id={applicant_id} | "
        f"query={query[:80]}"
    )

    # Embed the query using same model as ingestion — must match else similarity is meaningless
    query_embedding = embed_query(query)

    # Retrieve top-K chunks filtered to this applicant only
    chunks = query_similar_chunks(
        query_embedding=query_embedding,
        applicant_id=applicant_id
    )

    # Join chunk texts into one string for direct prompt injection
    context_text = "\n\n---\n\n".join([
        f"[Page {c['page_number']}] {c['text']}"
        for c in chunks
    ])

    logger.info(
        f"Retrieval complete | "
        f"applicant_id={applicant_id} | "
        f"chunks_retrieved={len(chunks)} | "
        f"top_score={chunks[0]['similarity_score'] if chunks else 'N/A'}"
    )

    return {
        "chunks": chunks,
        "query_used": query,
        "context_text": context_text
    }