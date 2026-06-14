import time
import chromadb
from chromadb.config import Settings as ChromaSettings
#from chromadb import Settings as ChromaSettings
from app.config import settings
from app.utils import get_logger

logger = get_logger(__name__)

# number of top result needed -centralised in config
TOP_K = settings.top_k_results

# ChromaDB client created once at module load 
# PersistentClient writes index to disk so vectors survive server restarts
_client = chromadb.PersistentClient(
    path=settings.chroma_persist_dir,
    settings=ChromaSettings(anonymized_telemetry=False)  # disable telemetry pings to ChromaDB servers
)


def get_or_create_collection() -> chromadb.Collection:
    """
    Returns the underwriting docs collection, creating it if it doesn't exist.
    ChromaDB collections are idempotent — calling get_or_create on an existing
    collection returns it unchanged, so this is safe to call multiple times.

    Returns:
        ChromaDB collection configured with cosine similarity
    """
    collection = _client.get_or_create_collection(
        name=settings.chroma_collection_name,
        metadata={"hnsw:space": "cosine"}  # cosine similarity — direction = meaning, not magnitude
    )

    logger.info(
        f"Collection ready | "
        f"name={settings.chroma_collection_name} | "
        f"existing_docs={collection.count()}"
    )

    return collection


def upsert_chunks(chunks: list[dict], applicant_id: str) -> None:
    """
    Stores embedded chunks in ChromaDB. Uses upsert not insert —
    if a chunk with the same ID already exists it is overwritten.
    This makes re-processing a document safe with no duplicates.

    Args:
        chunks: output from embedder.embed_chunks — each chunk must have
                chunk_id, text, page_number, token_count, embedding
        applicant_id: unique identifier for this applicant/case —
                      stored as metadata for filtered retrieval later

    Raises:
        ValueError: if chunks list is empty or embeddings are missing
    """
    if not chunks:
        raise ValueError("No chunks provided for upsert")

    # Fail fast — catch missing embeddings before sending to ChromaDB
    if "embedding" not in chunks[0]:
        raise ValueError(
            "Chunks must be embedded before upserting — run embed_chunks() first"
        )

    collection = get_or_create_collection()

    # ChromaDB expects four parallel lists — ids, embeddings, documents, metadatas
    # Each index i across all four lists describes one chunk
    ids = [f"{applicant_id}_chunk_{chunk['chunk_id']}" for chunk in chunks]

    embeddings = [chunk["embedding"] for chunk in chunks]

    documents = [chunk["text"] for chunk in chunks]

    metadatas = [
        {
            "applicant_id": applicant_id,
            "page_number": chunk["page_number"],
            "chunk_id": chunk["chunk_id"],
            "token_count": chunk["token_count"]
        }
        for chunk in chunks
    ]

    start_time = time.perf_counter()

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas
    )

    latency_ms = round((time.perf_counter() - start_time) * 1000)

    logger.info(
        f"Upsert complete | "
        f"applicant_id={applicant_id} | "
        f"chunks={len(chunks)} | "
        f"latency={latency_ms}ms | "
        f"total_in_collection={collection.count()}"
    )


def query_similar_chunks(
    query_embedding: list[float],
    applicant_id: str,
    top_k: int = TOP_K
) -> list[dict]:
    """
    Retrieves the top-K chunks most semantically similar to the query.
    Filters by applicant_id so each case only searches its own documents —
    prevents one applicant's data leaking into another's risk assessment.

    Args:
        query_embedding: vector from embedder.embed_query()
        applicant_id: filter results to this applicant's chunks only
        top_k: number of chunks to return (default from config)

    Returns:
        list of dicts with keys: text, applicant_id, page_number,
        chunk_id, token_count, similarity_score

    Raises:
        ValueError: if query_embedding is empty
    """
    if not query_embedding:
        raise ValueError("Query embedding cannot be empty")

    collection = get_or_create_collection()

    start_time = time.perf_counter()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where={"applicant_id": applicant_id},  # applicant isolation — never mix cases
        include=["documents", "metadatas", "distances"]
    )

    latency_ms = round((time.perf_counter() - start_time) * 1000)

    # ChromaDB returns nested lists — index [0] unwraps the single-query wrapper
    # distances are 0-1 where 0 = identical, 1 = completely different
    # convert to similarity score so higher = more similar (matches intuition)
    chunks = []
    for doc, metadata, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        chunks.append({
            "text": doc,
            "applicant_id": metadata["applicant_id"],
            "page_number": metadata["page_number"],
            "chunk_id": metadata["chunk_id"],
            "token_count": metadata["token_count"],
            "similarity_score": round(1 - distance, 4)  # convert distance → similarity
        })

    logger.info(
        f"Query complete | "
        f"applicant_id={applicant_id} | "
        f"top_k={top_k} | "
        f"latency={latency_ms}ms | "
        f"top_score={chunks[0]['similarity_score'] if chunks else 'N/A'}"
    )

    return chunks