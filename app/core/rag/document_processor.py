import fitz
import tiktoken
from pathlib import Path
from app.utils import clean_text, get_logger

logger = get_logger(__name__)

# Created once at module load — tiktoken encoder initialisation
# loads vocabulary files from disk, expensive to repeat per call
ENCODER = tiktoken.get_encoding("cl100k_base")


def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    """
    Extracts text from each page of a PDF file.
    For scanned PDFs where get_text() returns empty, an OCR fallback- AWS Textract 
    
    Args:
        pdf_path: path of PDF file

    Returns:
        list of dicts with keys: page_number, text
    """
    pages = []

    try:
        with fitz.open(pdf_path) as doc:
            for page_num in range(len(doc)):
                try:
                    page = doc[page_num]
                    text = page.get_text()

                    if not text.strip():
                        logger.warning(
                            f"Page {page_num + 1} yielded no text — "
                            f"possibly scanned or image-based, skipping"
                        )
                        continue

                    pages.append({
                        "page_number": page_num + 1,
                        "text": clean_text(text)
                    })

                except Exception as e:
                    logger.warning(f"Skipping page {page_num + 1}: {e}")
                    continue

    except fitz.FileDataError:
        # to take care in case contents are corrupted
        raise ValueError(f"Invalid or corrupted PDF: {pdf_path}")

    except Exception as e:
        raise RuntimeError(f"PDF processing failed: {e}") from e

    if not pages:
        raise ValueError(f"No text extracted from: {pdf_path}")

    return pages


def chunk_text(pages: list[dict], chunk_size: int = 512,
               overlap: int = 50) -> list[dict]:
    """
    Splits pages into overlapping token-based chunks ready for embedding.
    Uses tiktoken to split on tokens rather than words. Overlap prevents losing
    context at chunk boundaries.

    Args:
        pages: output from extract_text_from_pdf
        chunk_size: max tokens per chunk (default 512)
        overlap: tokens shared between adjacent chunks (default 50)

    Returns:
        list of chunk dicts with keys:
            chunk_id, text, page_number, token_count
    """
    if overlap >= chunk_size:
        raise ValueError(
            f"Overlap ({overlap}) must be less than chunk_size ({chunk_size})"
        )

    chunks = []
    chunk_id = 0

    for page in pages:
        try:
            tokens = ENCODER.encode(page["text"])
            page_number = page["page_number"]

            if not tokens:
                continue

            start = 0
            while start < len(tokens):
                end = start + chunk_size
                chunk_tokens = tokens[start:end]

                chunks.append({
                    "chunk_id": chunk_id,
                    "text": ENCODER.decode(chunk_tokens),
                    # a chunk containing too many unrelated facts produces a diluted vector that performs poorly in similarity search.
                    #insurance documents with medical terminology — the word to token ratio is unpredictable
                    "page_number": page_number,
                    "token_count": len(chunk_tokens)
                })

                chunk_id += 1
                # Advance by chunk_size minus overlap to create sliding window
                start += chunk_size - overlap

        except Exception as e:
            logger.warning(f"Skipping page {page.get('page_number')}: {e}")
            continue

    return chunks


def process_pdf(pdf_path: str) -> list[dict]:
    """
    Main entry point — processes a PDF into embedding-ready chunks.

    Validates the file at the boundary, extracts and cleans text
    page by page, then splits into overlapping token-based chunks.

    Args:
        pdf_path: path to the PDF file

    Returns:
        list of chunk dicts ready for embed_chunks()
    Raises:
        ValueError: if file missing, not a PDF, or no text extracted
    """
    path = Path(pdf_path)

    # Fail fast — validate at the boundary before any processing
    if not path.exists():
        raise ValueError(f"File does not exist: {pdf_path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError(f"File is not a PDF: {pdf_path}")

    logger.info(f"Processing PDF: {pdf_path} ({path.stat().st_size} bytes)")

    pages = extract_text_from_pdf(pdf_path)
    logger.info(f"Extracted {len(pages)} pages")

    chunks = chunk_text(pages)
    logger.info(
        f"Created {len(chunks)} chunks from {len(pages)} pages | "
        f"chunk_size=512 tokens | overlap=50 tokens"
    )

    return chunks