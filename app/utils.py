import re
import logging

# Pricing constants — update here when provider pricing changes

# Sonnet pricing per million tokens
SONNET_INPUT_COST_PER_M_TOKENS  = 3.0
SONNET_OUTPUT_COST_PER_M_TOKENS = 15.0

# Haiku pricing per million tokens — 12x cheaper than Sonnet for evaluation
HAIKU_INPUT_COST_PER_M_TOKENS  = 0.25
HAIKU_OUTPUT_COST_PER_M_TOKENS = 1.25

# OpenAI text-embedding-3-small pricing per million tokens
EMBEDDING_COST_PER_M_TOKENS = 0.02


def get_logger(name: str) -> logging.Logger:
    """
    Returns a configured logger for the given module.

    Usage in any file:
        from app.utils import get_logger
        logger = get_logger(__name__)

    Args:
        name: module name — always pass __name__

    Returns:
        configured Logger instance
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if get_logger is called multiple times for the same module
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

    return logger


def clean_text(text: str) -> str:
    """
    Cleans raw text extracted from PDFs or other sources.
    Removes noise that degrades embedding quality.

    Reusable across the entire pipeline — document processor,
    LLM response cleaning, evaluation inputs.

    Args:
        text: raw text to clean

    Returns:
        cleaned text string
    """
    # Drop undecodable bytes rather than letting them corrupt downstream embeddings
    text = text.encode("utf-8", errors="ignore").decode("utf-8")

    # Matches "Page 1", "page 23 of 45", "PAGE 1 OF 10"
    text = re.sub(r'\bpage\s+\d+\s*(of\s*\d+)?\b', '', text, flags=re.IGNORECASE)

    # Matches "- 1 -", "- 23 -" but not date separators like 1979-03-15
    text = re.sub(r'(?<!\d)-\s*\d{1,3}\s*-(?!\d)', '', text)

    text = re.sub(r'\bCONFIDENTIAL\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\bDRAFT\b', '', text, flags=re.IGNORECASE)

    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = text.strip()

    return text


def calculate_claude_cost(
    tokens_in: int,
    tokens_out: int,
    model: str = None
) -> float:
    """
    Estimates cost in USD for a Claude API call.
    Uses model-specific pricing — defaults to Sonnet pricing.

    Args:
        tokens_in:  number of input tokens consumed
        tokens_out: number of output tokens generated
        model:      Claude model used — determines pricing tier

    Returns:
        estimated cost in USD rounded to 6 decimal places
    """
    if model and "haiku" in model.lower():
        input_rate  = HAIKU_INPUT_COST_PER_M_TOKENS
        output_rate = HAIKU_OUTPUT_COST_PER_M_TOKENS
    else:
        input_rate  = SONNET_INPUT_COST_PER_M_TOKENS
        output_rate = SONNET_OUTPUT_COST_PER_M_TOKENS

    input_cost  = (tokens_in  / 1_000_000) * input_rate
    output_cost = (tokens_out / 1_000_000) * output_rate
    return round(input_cost + output_cost, 6)


def calculate_embedding_cost(total_tokens: int) -> float:
    """
    Estimates cost in USD for an OpenAI embedding API call.

    Args:
        total_tokens: total tokens embedded

    Returns:
        estimated cost in USD rounded to 6 decimal places
    """
    return round((total_tokens / 1_000_000) * EMBEDDING_COST_PER_M_TOKENS, 8)



def strip_markdown_json(text: str) -> str:
    """
    Strips markdown code fences from LLM JSON responses.
    Claude sometimes wraps JSON in ```json ... ``` blocks
    """
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return text.strip()