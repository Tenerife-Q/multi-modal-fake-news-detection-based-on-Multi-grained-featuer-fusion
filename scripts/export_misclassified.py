import json
import os
import sys
from argparse import Namespace

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from trainMMFN import build_dataloader
from MMFN import MultiModal


def main():
    args = Namespace(batch_size=32, use_entity_enrich=True, dataset="gossip")

    model = MultiModal()
    ckpt_path = "ckpt/mmfn_base/best_model.pth"
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state)
    model = model.cuda() if torch.cuda.is_available() else model.cpu()
    model.eval()

    _, test_loader, dataset_name = build_dataloader(args)
    dataset = test_loader.dataset

    mistakes = []
    total = 0
    correct = 0

    with torch.no_grad():
        for batch in test_loader:
            (
                input_ids, attention_mask, token_type_ids,
                entity_input_ids, entity_attention_mask, entity_token_type_ids,
                image, imageclip, textclip, labels, sample_indices
            ) = batch

            if torch.cuda.is_available():
                input_ids = input_ids.cuda()
                attention_mask = attention_mask.cuda()
                token_type_ids = token_type_ids.cuda()
                entity_input_ids = entity_input_ids.cuda()
                entity_attention_mask = entity_attention_mask.cuda()
                entity_token_type_ids = entity_token_type_ids.cuda()
                image = image.cuda()
                imageclip = imageclip.cuda()
                textclip = textclip.cuda()
                labels = labels.cuda()

            from gossipcop_dataset import clipmodel
            image_clip = clipmodel.encode_image(imageclip)
            text_clip = clipmodel.encode_text(textclip)

            logits, _ = model(
                input_ids, attention_mask, token_type_ids,
                image, text_clip, image_clip,
                labels=None,
                dataset_name=dataset_name,
                entity_input_ids=entity_input_ids,
                entity_attention_mask=entity_attention_mask,
                entity_token_type_ids=entity_token_type_ids,
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
                    "image_path": rec.get("image_path", ""),
                    "title_or_summary": rec.get("sum_content", "")[:300],
                    "content_preview": rec.get("content", "")[:500],
                })

    acc = correct / total if total else 0.0
    out = {
        "checkpoint": ckpt_path,
        "total_samples": total,
        "correct_samples": correct,
        "accuracy": acc,
        "misclassified_count": len(mistakes),
        "misclassified_examples": mistakes,
    }

    with open("gossip/misclassified_report.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(json.dumps({
        "accuracy": acc,
        "misclassified_count": len(mistakes),
        "report_path": "gossip/misclassified_report.json",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
