"""
api_utils.py

Shared retry logic for Gemini API calls..

"""


import re
import time
from google.genai import errors as genai_errors

_last_call_time: dict[str, float] = {}


def _pace(bucket: str, min_interval_seconds: float):
    now = time.time()
    last = _last_call_time.get(bucket, 0)
    elapsed = now - last

    if elapsed < min_interval_seconds:
        time.sleep(min_interval_seconds - elapsed)

    _last_call_time[bucket] = time.time()


def call_with_retry(api_call, max_retries: int = 10, default_wait: float = 60,
                     bucket: str | None = None, min_interval_seconds: float = 0):
    """
    Calls api_call() (pass a zero-argument function, e.g. via lambda),
    retrying automatically on known transient errors: rate limits, server
    overload, and network/DNS failures. Optionally paces calls under a
    named bucket to proactively stay under a known rate limit.
    """
    if bucket is not None and min_interval_seconds > 0:
        _pace(bucket, min_interval_seconds)

    for attempt in range(max_retries):
        try:
            return api_call()

        except genai_errors.APIError as e:
            error_str = str(e)
            label = f"[{bucket or 'unlabeled'}]"

            if "RESOURCE_EXHAUSTED" in error_str:
                wait_seconds = _extract_wait_seconds(error_str, default_wait)
                print(f"  {label} Rate limit hit. Waiting {wait_seconds:.0f}s before retrying "
                      f"(attempt {attempt + 1}/{max_retries})...")
            elif "UNAVAILABLE" in error_str or "503" in error_str:
                wait_seconds = default_wait / 2
                print(f"  {label} Model temporarily unavailable (server overload). Waiting "
                      f"{wait_seconds:.0f}s before retrying (attempt {attempt + 1}/{max_retries})...")
            else:
                raise
            time.sleep(wait_seconds)

        except Exception as e:
            error_module = type(e).__module__
            error_str = str(e)
            is_network_error = (
                error_module.startswith("httpx") or
                error_module.startswith("httpcore") or
                any(sig in error_str for sig in [
                    "nodename nor servname", "Name or service not known",
                    "Network is unreachable", "getaddrinfo",
                ])
            )
            if not is_network_error:
                raise

            label = f"[{bucket or 'unlabeled'}]"
            wait_seconds = 10
            print(f"  {label} Network error ({type(e).__name__}: {error_str}). Waiting "
                  f"{wait_seconds}s before retrying (attempt {attempt + 1}/{max_retries})...")
            time.sleep(wait_seconds)

    raise RuntimeError(f"Exceeded max retries ({max_retries}) due to transient errors.")


def _extract_wait_seconds(error_text: str, default: float) -> float:
    match = re.search(r"retry in (\d+(?:\.\d+)?)s", error_text)
    return float(match.group(1)) + 2 if match else default

