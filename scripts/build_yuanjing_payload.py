import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from blockchain_bridge import build_integration_bundle, dumps_bundle


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build yuanjing-core compatible payloads from MMFN prediction outputs."
    )
    parser.add_argument("--dataset", required=True, choices=["weibo", "gossip"], help="Dataset name")
    parser.add_argument("--image-path", required=True, help="Absolute or relative image path")
    parser.add_argument("--pred-label", required=True, type=int, help="Predicted class index from MMFN")
    parser.add_argument("--confidence", required=True, type=float, help="Prediction confidence in [0, 1]")
    parser.add_argument("--checkpoint", help="Checkpoint path used to derive a model governance hash")
    parser.add_argument("--description", help="Model description for yuanjing-core /model/register")
    parser.add_argument("--source", default="mmfn", help="Source field for yuanjing-core /prove")
    parser.add_argument("--sample-id", help="Optional local sample identifier")
    parser.add_argument("--external-knowledge", default="", help="Optional external knowledge text for local hashing")
    parser.add_argument(
        "--activated-prompts",
        default="",
        help="Comma-separated prompt ids for local bookkeeping, e.g. 1,2,99",
    )
    return parser.parse_args()


def parse_prompt_ids(raw_value: str):
    if not raw_value.strip():
        return []
    return [int(part.strip()) for part in raw_value.split(",") if part.strip()]


def main():
    args = parse_args()
    bundle = build_integration_bundle(
        dataset=args.dataset,
        image_path=args.image_path,
        predicted_label=args.pred_label,
        confidence=args.confidence,
        checkpoint_path=args.checkpoint,
        source=args.source,
        sample_id=args.sample_id,
        external_knowledge=args.external_knowledge,
        description=args.description,
        activated_prompts=parse_prompt_ids(args.activated_prompts),
    )
    print(dumps_bundle(bundle))


if __name__ == "__main__":
    main()
