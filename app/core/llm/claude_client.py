
import time
import anthropic
from app.config import settings
from app.utils import get_logger, calculate_claude_cost

logger = get_logger(__name__)

# Single client instance created once at module load — singleton pattern
# Avoids overhead of creating new HTTP connections on every API call
_client = anthropic.Anthropic(api_key=settings.anthropic_api_key)


def call_claude(
    prompt: str,
    system_prompt: str = "You are a helpful AI assistant.",
    max_tokens: int = 1000,
    temperature: float = 0.0,
    model: str = None
) -> dict:
    """
    Makes a single call to the Claude API and returns the response
    with a trace for observability.
    
    Args:
        prompt: user message to send to Claude
        system_prompt: sets Claude's persona and task constraints
        max_tokens: maximum tokens in response
        temperature: output randomness (0.0=deterministic, 1.0=creative)
        model: Claude model to use — defaults to settings.claude_model
            pass settings.claude_judge_model for evaluation calls

    Returns:
        dict with keys:
            - text (str): Claude's response as plain text
            - trace (dict): observability data with keys:
                - model (str): model name used
                - tokens_in (int): input tokens consumed
                - tokens_out (int): output tokens generated
                - latency_ms (int): total API call duration in milliseconds
                - cost_usd (float): estimated cost in US dollars
    """
    # Fail fast — validate inputs at the boundary before making an expensive API call with bad data
    if not prompt or not prompt.strip():
        raise ValueError("Prompt cannot be empty")

    if not 0.0 <= temperature <= 1.0:
        raise ValueError(
            f"Temperature must be between 0.0 and 1.0, got {temperature}"
        )

    if max_tokens < 1:
        raise ValueError(
            f"max_tokens must be at least 1, got {max_tokens}"
        )

    # use provided model or fall back to settings
    model_to_use = model or settings.claude_model

    # perf_counter is more precise than time.time() 
    start_time = time.perf_counter()
    
    try:
        response = _client.messages.create(
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=[
                {"role": "user", "content": prompt}
            ],
            model=model_to_use
        )

    except anthropic.AuthenticationError as e:
        # API key is invalid — retrying will not help
        logger.error(f"Invalid Anthropic API key: {e}")
        raise

    except anthropic.RateLimitError as e:
        # Calling too fast — caller should back off and retry
        logger.warning(f"Anthropic rate limit hit — back off and retry: {e}")
        raise

    except anthropic.APIConnectionError as e:
        # Network issue — may resolve on retry
        logger.error(f"Network connection to Anthropic failed: {e}")
        raise

    except anthropic.APIStatusError as e:
        logger.error(f"Anthropic API returned error status {e.status_code}: {e}")
        raise

    except Exception as e:
        # Unexpected error — wrap with context but preserve original cause
        logger.error(f"Unexpected error during Claude API call: {e}")
        raise RuntimeError(f"Claude API call failed: {e}") from e

    latency_ms = round((time.perf_counter() - start_time) * 1000)

    trace = {"model": model_to_use,
        "tokens_in": response.usage.input_tokens,
        "tokens_out": response.usage.output_tokens,
        "latency_ms": latency_ms,
        "cost_usd": calculate_claude_cost(response.usage.input_tokens,
         response.usage.output_tokens,model=model_to_use )}

    logger.info(
        f"Claude call complete | "
        f"tokens_in={trace['tokens_in']} | "
        f"tokens_out={trace['tokens_out']} | "
        f"latency={trace['latency_ms']}ms | "
        f"cost=${trace['cost_usd']}"
    )

    return {
        "text": response.content[0].text,
        "trace": trace
    }


