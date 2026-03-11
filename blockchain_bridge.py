import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


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


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_confidence(confidence: float) -> float:
    clipped = max(0.0, min(float(confidence), 1.0))
    return round(clipped, 6)


def label_to_verdict(dataset: str, predicted_label: int) -> bool:
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
class ModelRegistration:
    model_hash: str
    description: str

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, description: str) -> "ModelRegistration":
        return cls(model_hash=sha256_file(checkpoint_path), description=description)

    def to_api_payload(self) -> Dict[str, str]:
        return {
            "hash": self.model_hash,
            "description": self.description,
        }


@dataclass
class PredictionEvidence:
    dataset: str
    image_path: str
    predicted_label: int
    confidence: float
    checkpoint_path: Optional[str] = None
    source: str = "mmfn"
    sample_id: Optional[str] = None
    external_knowledge: str = ""
    prompt_pool_hash: str = ""
    activated_prompts: List[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.dataset = self.dataset.lower()
        self.image_path = str(Path(self.image_path))
        self.predicted_label = int(self.predicted_label)
        self.confidence = normalize_confidence(self.confidence)
        if not self.prompt_pool_hash and self.checkpoint_path:
            self.prompt_pool_hash = sha256_file(self.checkpoint_path)

    @property
    def verdict(self) -> bool:
        return label_to_verdict(self.dataset, self.predicted_label)

    @property
    def external_knowledge_hash(self) -> str:
        if not self.external_knowledge:
            return ""
        return sha256_text(self.external_knowledge)

    def to_prove_payload(self) -> Dict[str, Any]:
        return {
            "image_path": self.image_path,
            "verdict": self.verdict,
            "confidence": self.confidence,
            "source": self.source,
        }

    def to_local_record(self) -> Dict[str, Any]:
        record = asdict(self)
        record.update(
            {
                "verdict": self.verdict,
                "external_knowledge_hash": self.external_knowledge_hash,
                "prove_payload": self.to_prove_payload(),
            }
        )
        return record


def build_integration_bundle(
    dataset: str,
    image_path: str,
    predicted_label: int,
    confidence: float,
    checkpoint_path: Optional[str] = None,
    source: str = "mmfn",
    sample_id: Optional[str] = None,
    external_knowledge: str = "",
    description: Optional[str] = None,
    activated_prompts: Optional[List[int]] = None,
) -> Dict[str, Any]:
    evidence = PredictionEvidence(
        dataset=dataset,
        image_path=image_path,
        predicted_label=predicted_label,
        confidence=confidence,
        checkpoint_path=checkpoint_path,
        source=source,
        sample_id=sample_id,
        external_knowledge=external_knowledge,
        activated_prompts=activated_prompts or [],
    )

    bundle: Dict[str, Any] = {
        "prove": evidence.to_prove_payload(),
        "local_record": evidence.to_local_record(),
    }

    if checkpoint_path:
        registration = ModelRegistration.from_checkpoint(
            checkpoint_path=checkpoint_path,
            description=description or f"MMFN checkpoint: {Path(checkpoint_path).name}",
        )
        bundle["register_model"] = registration.to_api_payload()

    return bundle


def dumps_bundle(bundle: Dict[str, Any]) -> str:
    return json.dumps(bundle, ensure_ascii=False, indent=2, sort_keys=True)
