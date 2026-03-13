import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import requests

# Base URL for the yuanjing-core service (can be overridden via environment variable).
YUANJING_BASE_URL: str = os.environ.get("YUANJING_BASE_URL", "http://localhost:8080")

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
# HTTP client helpers for yuanjing-core
# ---------------------------------------------------------------------------

def health_check(base_url: str = YUANJING_BASE_URL, timeout: int = 5) -> bool:
    """Return True if the yuanjing-core service is reachable and healthy."""
    try:
        response = requests.get(f"{base_url}/health", timeout=timeout)
        return response.status_code == 200
    except requests.RequestException:
        return False


def submit_proof(
    payload: PredictionPayload,
    base_url: str = YUANJING_BASE_URL,
    timeout: int = 30,
) -> Dict[str, Any]:
    """Submit a prediction proof to the yuanjing-core service.

    Returns the JSON response body as a dict.
    Raises :class:`requests.HTTPError` on a non-2xx response.
    """
    response = requests.post(
        f"{base_url}/api/v1/proofs",
        json=payload.to_api_payload(),
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def submit_proof_with_retry(
    payload: PredictionPayload,
    max_retries: int = 3,
    backoff_seconds: float = 2.0,
    base_url: str = YUANJING_BASE_URL,
    timeout: int = 30,
) -> Dict[str, Any]:
    """Submit a prediction proof with exponential-backoff retry.

    Retries on :class:`requests.RequestException` (network errors, timeouts).
    Re-raises the last exception if all attempts are exhausted.
    ``max_retries`` must be at least 1.
    """
    if max_retries < 1:
        raise ValueError("max_retries must be >= 1")
    last_exc: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            return submit_proof(payload, base_url=base_url, timeout=timeout)
        except requests.RequestException as exc:
            last_exc = exc
            if attempt < max_retries:
                sleep_time = backoff_seconds * (2 ** (attempt - 1))
                time.sleep(sleep_time)
    raise last_exc  # type: ignore[misc]


def verify_audit(
    receipt_id: str,
    base_url: str = YUANJING_BASE_URL,
    timeout: int = 30,
) -> Dict[str, Any]:
    """Verify a previously submitted proof by its receipt ID.

    Returns the JSON verification result.
    Raises :class:`requests.HTTPError` on a non-2xx response.
    """
    response = requests.get(
        f"{base_url}/api/v1/proofs/{receipt_id}",
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()
