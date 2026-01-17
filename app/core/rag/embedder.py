import time
from openai import OpenAI, AuthenticationError, RateLimitError, APIConnectionError
from app.config import settings
from app.utils import get_logger, calculate_embedding_cost

logger = get_logger(__name__)

# Single client instance created once at module load — singleton pattern
_client = OpenAI(api_key=settings.openai_api_key)


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Adds an embedding vector to each chunk.

    Sends all chunk texts in a single batched API call — more efficient
    than embedding one chunk at a time since OpenAI billing is the same
    per token regardless of batch size.

    Args:
        chunks: output from document_processor.process_pdf

    Returns:
        same chunks with embedding key added — list of 1536 floats
    """
    # Fail fast — validate at boundary before expensive API call
    if not chunks:
        raise ValueError("No chunks provided for embedding")

    texts = [chunk["text"] for chunk in chunks]

    logger.info(f"Embedding {len(chunks)} chunks...")

    start_time = time.perf_counter()

    try:
        response = _client.embeddings.create(
            model=settings.embedding_model,
            input=texts
        )

    except AuthenticationError as e:
        # API key is invalid — retrying will not help
        logger.error(f"Invalid OpenAI API key: {e}")
        raise

    except RateLimitError as e:
        # insufficient_quota needs billing action — retrying will not help
        if "insufficient_quota" in str(e):
            logger.error(f"OpenAI quota exhausted — billing action required: {e}")
        else:
            logger.warning(f"OpenAI rate limit hit — back off and retry: {e}")
        raise

    except Exception as e:
        logger.error(f"Unexpected error during embedding: {e}")
        raise RuntimeError(f"Failed to embed chunks: {e}") from e

    latency_ms = round((time.perf_counter() - start_time) * 1000)

    # OpenAI returns embeddings in the same order as input texts
    # index i in response.data corresponds to index i in chunks
    for i, chunk in enumerate(chunks):
        chunk["embedding"] = response.data[i].embedding

    logger.info(
        f"Embedding complete | "
        f"chunks={len(chunks)} | "
        f"dimensions={len(response.data[0].embedding)} | "
        f"latency={latency_ms}ms | "
        f"chunk embed cost=${calculate_embedding_cost(response.usage.total_tokens)}"
    )

    return chunks


def embed_query(query: str) -> list[float]:
    """
    Embeds a single query string for similarity search.
    Called at retrieval time to convert a search query into a vector

    Args:
        query: search query string

    Returns:
        list of 1536 floats representing the query's semantic meaning

    Raises:
        ValueError: if query is empty
        RuntimeError: if embedding API call fails
    """
    # Fail fast — validate at boundary before API call
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")

    start_time = time.perf_counter()

    try:
        response = _client.embeddings.create(
            model=settings.embedding_model,
            input=[query]
        )

    except AuthenticationError as e:
        # API key is invalid — retrying will not help
        logger.error(f"Invalid OpenAI API key: {e}")
        raise

    except RateLimitError as e:
        # insufficient_quota needs billing action — retrying will not help
        if "insufficient_quota" in str(e):
            logger.error(f"OpenAI quota exhausted — billing action required: {e}")
        else:
            logger.warning(f"OpenAI rate limit hit — back off and retry: {e}")
        raise

    except Exception as e:
        logger.error(f"Unexpected error during query embedding: {e}")
        raise RuntimeError(f"Failed to embed query: {e}") from e

    latency_ms = round((time.perf_counter() - start_time) * 1000)

    # Use API-reported token count for cost — more accurate than re-encoding locally
    logger.info(
        f"Query embedding complete | "
        f"latency={latency_ms}ms | "
        f"query embed cost=${calculate_embedding_cost(response.usage.total_tokens)}"
    )

    return response.data[0].embedding