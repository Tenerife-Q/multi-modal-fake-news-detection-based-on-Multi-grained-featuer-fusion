from dataclasses import dataclass
from pathlib import Path
from typing import Dict


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
