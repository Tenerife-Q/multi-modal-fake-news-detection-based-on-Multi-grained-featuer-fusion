import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import requests


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BLOCKCHAIN_ENDPOINT = os.getenv("YUANJING_API_ENDPOINT", "http://localhost:3000")
DEFAULT_TIMEOUT = float(os.getenv("YUANJING_API_TIMEOUT", "10.0"))

DATASET_LABEL_TO_VERDICT = {
    "weibo": {
        0: False,
        1: True,
    },
    "gossip": {
        0: True,
        1: False,
    },
}

def normalize_confidence(confidence: float) -> float:
    """Clip confidence into [0.0, 1.0] and round to 6 decimals for stable payloads."""
    clipped = max(0.0, min(float(confidence), 1.0))
    return round(clipped, 6)


def label_to_verdict(dataset: str, predicted_label: int) -> bool:
    """Map MMFN class ids to a boolean verdict understood by the blockchain side."""
    dataset_key = dataset.lower()
    if dataset_key not in DATASET_LABEL_TO_VERDICT:
        raise ValueError(f"Unsupported dataset '{dataset}'. Expected one of {sorted(DATASET_LABEL_TO_VERDICT)}.")

    try:
        return DATASET_LABEL_TO_VERDICT[dataset_key][int(predicted_label)]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported label '{predicted_label}' for dataset '{dataset}'."
        ) from exc


@dataclass
class PredictionPayload:
    """Minimal phase-one payload that standardizes MMFN prediction semantics for later API wiring."""
    dataset: str
    image_path: str
    predicted_label: int
    confidence: float
    source: str = "mmfn"

    def __post_init__(self) -> None:
        self.dataset = self.dataset.lower()
        self.image_path = str(Path(self.image_path))
        self.predicted_label = int(self.predicted_label)
        self.confidence = normalize_confidence(self.confidence)

    @property
    def verdict(self) -> bool:
        return label_to_verdict(self.dataset, self.predicted_label)

    def to_api_payload(self) -> Dict[str, object]:
        return {
            "image_path": self.image_path,
            "verdict": self.verdict,
            "confidence": self.confidence,
            "source": self.source,
        }


# ---------------------------------------------------------------------------
# HTTP client functions
# ---------------------------------------------------------------------------

def submit_proof(payload: PredictionPayload, timeout: Optional[float] = None) -> dict:
    """
    Call the yuanjing-core /prove endpoint to submit a prediction proof.

    Args:
        payload: A PredictionPayload instance carrying the inference result.
        timeout: Request timeout in seconds. Falls back to DEFAULT_TIMEOUT.

    Returns:
        JSON dict returned by the API (contains receipt_id, etc.).

    Raises:
        requests.HTTPError: API returned a non-2xx status code.
        requests.Timeout: Request timed out.
        requests.ConnectionError: Could not reach the service.
    """
    url = f"{BLOCKCHAIN_ENDPOINT}/prove"
    resp = requests.post(
        url,
        json=payload.to_api_payload(),
        timeout=timeout or DEFAULT_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


def verify_audit(receipt_id: str, timeout: Optional[float] = None) -> dict:
    """
    Call the yuanjing-core /audit/{receipt_id} endpoint to verify a proof.

    Args:
        receipt_id: The proof receipt ID returned by submit_proof.
        timeout: Request timeout in seconds. Falls back to DEFAULT_TIMEOUT.

    Returns:
        JSON dict returned by the API (contains verification status).

    Raises:
        requests.HTTPError: API returned a non-2xx status code.
        requests.Timeout: Request timed out.
        requests.ConnectionError: Could not reach the service.
    """
    url = f"{BLOCKCHAIN_ENDPOINT}/audit/{receipt_id}"
    resp = requests.get(url, timeout=timeout or DEFAULT_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def health_check(timeout: float = 5.0) -> bool:
    """
    Check whether the yuanjing-core service is reachable and healthy.

    Args:
        timeout: Request timeout in seconds.

    Returns:
        True if the service responds with HTTP 200, False otherwise.
    """
    try:
        url = f"{BLOCKCHAIN_ENDPOINT}/health"
        resp = requests.get(url, timeout=timeout)
        return resp.status_code == 200
    except requests.RequestException:
        return False


def submit_proof_with_retry(
    payload: PredictionPayload,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    timeout: Optional[float] = None,
) -> dict:
    """
    Call submit_proof with exponential back-off retry on transient errors.

    Retries are attempted only for requests.Timeout and
    requests.ConnectionError. Any other exception (e.g. HTTP 4xx/5xx) is
    propagated immediately.

    Args:
        payload: A PredictionPayload instance carrying the inference result.
        max_retries: Maximum number of attempts (default: 3).
        retry_delay: Base delay in seconds between retries; doubles each time.
        timeout: Per-request timeout in seconds. Falls back to DEFAULT_TIMEOUT.

    Returns:
        JSON dict returned by the API on success.

    Raises:
        requests.Timeout or requests.ConnectionError: If all retries are
            exhausted without a successful response.
    """
    last_exception: Exception = RuntimeError("No attempts made")
    for attempt in range(max_retries):
        try:
            return submit_proof(payload, timeout=timeout)
        except (requests.Timeout, requests.ConnectionError) as exc:
            last_exception = exc
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2 ** attempt)
                logging.warning(
                    "submit_proof retry %d/%d after %.1fs: %s",
                    attempt + 1,
                    max_retries,
                    wait_time,
                    exc,
                )
                time.sleep(wait_time)
    raise last_exception
