import json
import yaml
from pathlib import Path
from app.core.llm.claude_client import call_claude
from app.utils import get_logger, strip_markdown_json, calculate_claude_cost

logger = get_logger(__name__)

# Load prompts once at module load — avoids re-reading file on every call
_PROMPTS_PATH = Path(__file__).parent.parent / "prompts" / "prompts.yaml"
with open(_PROMPTS_PATH, "r") as f:
    _PROMPTS = yaml.safe_load(f)

# Extraction prompt config — version tracked for observability
_PROMPT_CFG = _PROMPTS["extraction_prompt"]
PROMPT_VERSION = _PROMPT_CFG["version"]


def extract_application_data(document_text: str) -> dict:
    """
    LLM call #1 in the pipeline — extracts structured fields from raw PDF text.

    Uses Claude to parse unstructured insurance application text into a
    validated JSON object. Temperature is 0.0 for deterministic extraction —
    same document should always produce same structured output.

    Args:
        document_text: raw cleaned text from document_processor.process_pdf

    Returns:
        dict with keys:
            - data (dict): extracted applicant fields
            - prompt_version (str): which prompt version produced this output
            - trace (dict): latency, tokens, cost from claude_client

    Raises:
        ValueError: if document_text is empty or LLM returns unparseable JSON
        RuntimeError: if Claude API call fails
    """
    # Fail fast — validate at boundary before expensive API call
    if not document_text or not document_text.strip():
        raise ValueError("Document text cannot be empty")

    # Fill prompt template with real document text
    user_prompt = _PROMPT_CFG["user"].format(document_text=document_text)
    system_prompt = _PROMPT_CFG["system"]

    logger.info(
        f"Extracting application data | "  
        f"prompt_version={PROMPT_VERSION} | "
        f"doc_length={len(document_text)} chars"
    )

    response = call_claude(
        prompt=user_prompt,
        system_prompt=system_prompt,
        max_tokens=1000,
        temperature=0.0  # deterministic — same input must give same extraction
    )

    # Parse LLM response as JSON
    raw_text = strip_markdown_json(response["text"])

    try:
        extracted_data = json.loads(raw_text)

    except json.JSONDecodeError as e:
        # LLM returned malformed JSON — log raw response for debugging
        logger.error(
            f"LLM returned unparseable JSON | "
            f"prompt_version={PROMPT_VERSION} | "
            f"raw_response={raw_text[:200]}"
        )
        raise ValueError(
            f"Extraction failed — LLM did not return valid JSON: {e}"
        ) from e

    logger.info(
        f"Extraction complete | "
        f"prompt_version={PROMPT_VERSION} | "
        f"fields_extracted={list(extracted_data.keys())}"
    )

    return {
        "data": extracted_data,
        "prompt_version": PROMPT_VERSION,
        "trace": response["trace"]
    }