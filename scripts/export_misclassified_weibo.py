import argparse
import json
import os
import sys
from argparse import Namespace

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import trainMMFN as runner
from MMFN import MultiModal


def parse_args():
    parser = argparse.ArgumentParser(description="Export misclassified Weibo samples")
    parser.add_argument("--ckpt", type=str, default="ckpt/mmfn_base/best_model.pth", help="Path to model checkpoint")
    parser.add_argument("--batch_size", type=int, default=32, help="Evaluation batch size")
    parser.add_argument("--out", type=str, default="weibo/misclassified_report.json", help="Output JSON path")
    parser.add_argument("--use_entity_enrich", action="store_true", default=True,
                        help="Keep consistent with trainMMFN settings")
    return parser.parse_args()


def main():
    args = parse_args()

    loader_args = Namespace(
        batch_size=args.batch_size,
        use_entity_enrich=args.use_entity_enrich,
        dataset="weibo"
    )

    _, test_loader, dataset_name = runner.build_dataloader(loader_args)

    model = MultiModal()
    state = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(state)
    model = model.cuda() if torch.cuda.is_available() else model.cpu()
    model.eval()

    dataset = test_loader.dataset
    mistakes = []
    total = 0
    correct = 0

    with torch.no_grad():
        for batch in test_loader:
            (input_ids, attention_mask, token_type_ids,
             entity_input_ids, entity_attention_mask, entity_token_type_ids,
             image, imageclip, textclip, labels, sample_indices) = runner.unpack_batch(batch)

            input_ids = runner.to_var(input_ids)
            attention_mask = runner.to_var(attention_mask)
            token_type_ids = runner.to_var(token_type_ids)
            image = runner.to_var(image)
            imageclip = runner.to_var(imageclip)
            textclip = runner.to_var(textclip)
            labels = runner.to_var(labels)

            image_clip = runner.ACTIVE_CLIPMODEL.encode_image(imageclip)
            text_clip = runner.ACTIVE_CLIPMODEL.encode_text(textclip)

            logits, _ = model(
                input_ids, attention_mask, token_type_ids,
                image, text_clip, image_clip,
                labels=None,
                dataset_name=dataset_name,
                entity_input_ids=None,
                entity_attention_mask=None,
                entity_token_type_ids=None,
            )

            preds = logits.argmax(1)
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()

            wrong_mask = preds.ne(labels).detach().cpu()
            wrong_indices = sample_indices[wrong_mask].tolist()
            wrong_preds = preds.detach().cpu()[wrong_mask].tolist()
            wrong_labels = labels.detach().cpu()[wrong_mask].tolist()

            for idx, pred, gt in zip(wrong_indices, wrong_preds, wrong_labels):
                rec = dataset.label_dict[int(idx)]
                mistakes.append({
                    "dataset_index": int(idx),
                    "true_label": int(gt),
                    "pred_label": int(pred),
                    "images": rec.get("images", ""),
                    "title_or_summary": rec.get("sum_content", "")[:300],
                    "content_preview": rec.get("content", "")[:500],
                })

    acc = correct / total if total else 0.0
    out = {
        "dataset": "weibo",
        "checkpoint": args.ckpt,
        "total_samples": total,
        "correct_samples": correct,
        "accuracy": acc,
        "misclassified_count": len(mistakes),
        "misclassified_examples": mistakes,
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(json.dumps({
        "accuracy": acc,
        "misclassified_count": len(mistakes),
        "report_path": args.out,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
