"""
Shared utility: retry a callable on Mistral 429 rate-limit errors
with exponential backoff.
"""
import time
import httpx


def call_with_retry(fn, *args, max_retries: int = 5, base_wait: int = 5, **kwargs):
    """
    Call fn(*args, **kwargs). On a 429 rate-limit error, wait and retry
    with exponential backoff (5s → 10s → 20s → 40s → 80s).
    Raises on any other exception or after max_retries exhausted.
    """
    for attempt in range(1, max_retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            err_str = str(e).lower()
            is_429 = False

            if isinstance(e, httpx.HTTPStatusError) and e.response.status_code == 429:
                is_429 = True
            elif "429" in err_str or "rate limit" in err_str or "rate_limited" in err_str:
                is_429 = True

            if is_429 and attempt < max_retries:
                wait = base_wait * (2 ** (attempt - 1))   # 5, 10, 20, 40, 80
                print(f"  ⚠️  Mistral rate limit (429) hit — waiting {wait}s before retry {attempt}/{max_retries}…")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(f"Mistral API still rate-limiting after {max_retries} retries.")

