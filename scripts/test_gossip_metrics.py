import argparse
import json
import os
import sys
from argparse import Namespace

import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import trainMMFN as runner
from MMFN import MultiModal


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Gossip test set and print accuracy/precision/recall/f1")
    parser.add_argument("--ckpt", type=str, default="ckpt/mmfn_base/best_model.pth", help="Checkpoint path")
    parser.add_argument("--batch_size", type=int, default=32, help="Test batch size")
    parser.add_argument("--out", type=str, default="gossip/test_metrics.json", help="Output json path")
    parser.add_argument("--use_entity_enrich", action="store_true", default=True,
                        help="Keep consistent with train settings")
    return parser.parse_args()


def main():
    args = parse_args()

    loader_args = Namespace(
        batch_size=args.batch_size,
        use_entity_enrich=args.use_entity_enrich,
        dataset="gossip"
    )
    _, test_loader, dataset_name = runner.build_dataloader(loader_args)

    model = MultiModal()
    state = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(state)
    model = model.cuda() if torch.cuda.is_available() else model.cpu()
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for batch in test_loader:
            (input_ids, attention_mask, token_type_ids,
             entity_input_ids, entity_attention_mask, entity_token_type_ids,
             image, imageclip, textclip, labels, sample_indices) = runner.unpack_batch(batch)

            input_ids = runner.to_var(input_ids)
            attention_mask = runner.to_var(attention_mask)
            token_type_ids = runner.to_var(token_type_ids)
            entity_input_ids = runner.to_var(entity_input_ids)
            entity_attention_mask = runner.to_var(entity_attention_mask)
            entity_token_type_ids = runner.to_var(entity_token_type_ids)
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
                entity_input_ids=entity_input_ids,
                entity_attention_mask=entity_attention_mask,
                entity_token_type_ids=entity_token_type_ids,
            )

            preds = logits.argmax(1)
            y_true.extend(labels.detach().cpu().tolist())
            y_pred.extend(preds.detach().cpu().tolist())

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    conf = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()

    result = {
        "dataset": "gossip",
        "label_mapping": {
            "0": "real_news",
            "1": "fake_news"
        },
        "accuracy": float(acc),
        "metrics_by_class": {
            "real_news(label=0)": {
                "precision": float(precision[0]),
                "recall": float(recall[0]),
                "f1": float(f1[0]),
                "support": int(support[0])
            },
            "fake_news(label=1)": {
                "precision": float(precision[1]),
                "recall": float(recall[1]),
                "f1": float(f1[1]),
                "support": int(support[1])
            }
        },
        "confusion_matrix_labels_0_1": conf,
        "checkpoint": args.ckpt,
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
