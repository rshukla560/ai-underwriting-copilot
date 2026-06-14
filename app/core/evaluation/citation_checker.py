from app.core.evaluation.schemas import CitationResult
from app.utils import get_logger

logger = get_logger(__name__)

# minimum ratio of excerpt keywords that must appear in source chunks
KEYWORD_MATCH_THRESHOLD = 0.5


def check_citations(recommendation: dict, chunks: list[dict]) -> CitationResult:
    """
    Verifies each citation excerpt exists in retrieved source chunks.
    Rule-based — no LLM call needed.

    LLM generates excerpts claiming they are exact text from source.
    We verify those excerpts against real chunk text.
    Claim verification is faithfulness.py job — not handled here.

    Args:
        recommendation: output from recommender.generate_recommendation
        chunks: list of chunk dicts from vector_store

    Returns:
        CitationResult with accuracy score and lists of valid/invalid citations
    """
    if not recommendation:
        raise ValueError("Recommendation cannot be empty")

    citations = recommendation.get("citations", [])

    if not citations:
        logger.warning("No citations found in recommendation")
        return CitationResult(
            citation_accuracy=0.0,
            valid_citations=[],
            invalid_citations=[]
        )

    valid_citations = []
    invalid_citations = []

    # join all chunk texts once — reused for every citation
    all_text = " ".join([c["text"].lower() for c in chunks])

    for citation in citations:
        excerpt = citation.get("excerpt", "")
        claim = citation.get("claim", "unknown claim")

        # no excerpt — LLM did not provide text to verify
        if not excerpt:
            invalid_citations.append(f"no excerpt for: {claim}")
            logger.warning(f"No excerpt | claim={claim}")
            continue

        # extract keywords from excerpt
        keywords = [word.lower() for word in excerpt.split() if len(word) > 3]

        if not keywords:
            invalid_citations.append(f"no verifiable keywords: {excerpt}")
            continue

        # check if majority of excerpt keywords found in real chunks
        matches = [k for k in keywords if k in all_text]
        match_ratio = len(matches)/len(keywords)
        found = match_ratio >= KEYWORD_MATCH_THRESHOLD

        if found:
            valid_citations.append(excerpt)
        else:
            invalid_citations.append(f"not in source:{excerpt}")
            logger.warning(
                f"Excerpt not verified | "
                f"excerpt= {excerpt} | "
                f"match_ratio={round(match_ratio, 2)}"
            )

    total = len(valid_citations) + len(invalid_citations)
    score = round(len(valid_citations)/total, 4) if total > 0 else 0.0

    logger.info(
        f"Citation check | "
        f"score={score} | "
        f"valid={len(valid_citations)} | "
        f"invalid={len(invalid_citations)}"
    )

    return CitationResult(
        citation_accuracy=score,
        valid_citations=valid_citations,
        invalid_citations=invalid_citations
    )