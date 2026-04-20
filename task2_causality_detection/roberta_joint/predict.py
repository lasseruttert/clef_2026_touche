import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from task2_causality_detection.roberta_joint import config as C
from task2_causality_detection.roberta_joint.model import JointRoBERTa
from task2_causality_detection.roberta_joint.data import (
    bio_to_spans, load_detection_records, load_extraction_records,
    load_identification_records, make_collate_bio, make_collate_cls, make_loader,
)
from task2_causality_detection.common.pretty import quiet_transformers
from transformers import AutoTokenizer


def _predict_cls(model, task_id: int, records, collate_fn, device) -> list:
    loader = make_loader(records, collate_fn, C.BATCH_SIZE * 2, shuffle=False)
    all_preds = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            inp = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            out = model(task_id=task_id, **inp)
            all_preds.extend(out.logits.argmax(-1).cpu().tolist())
    return all_preds


def _predict_st2(model, ext_records, device, batch_size: int = 32) -> list:
    pad_id = 1
    all_spans = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(ext_records), batch_size):
            batch_recs = ext_records[i : i + batch_size]
            max_len = max(len(r["input_ids"]) for r in batch_recs)
            input_ids = torch.tensor([
                r["input_ids"] + [pad_id] * (max_len - len(r["input_ids"]))
                for r in batch_recs
            ]).to(device)
            attention_mask = torch.tensor([
                r["attention_mask"] + [0] * (max_len - len(r["attention_mask"]))
                for r in batch_recs
            ]).to(device)
            out = model(task_id=1, input_ids=input_ids, attention_mask=attention_mask)
            logits = out.logits.cpu().numpy()
            for j, rec in enumerate(batch_recs):
                n = len(rec["input_ids"])
                pred_ids = logits[j, :n].argmax(-1).tolist()
                pred_ids = [
                    -100 if rec["labels"][k] == -100 else pred_ids[k]
                    for k in range(n)
                ]
                spans = bio_to_spans(rec["offset_mapping"], pred_ids)
                all_spans.append([[s, e] for s, e in spans])
    return all_spans


def _write_jsonl(path: Path, rows: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def main():
    quiet_transformers()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    best_dir = C.OUTPUT_DIR / "best"
    ckpt = torch.load(best_dir / "model.pt", map_location="cpu")
    tokenizer = AutoTokenizer.from_pretrained(str(best_dir))

    model = JointRoBERTa(C.MODEL_NAME)
    model.resize_token_embeddings(ckpt["vocab_size"])
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)

    pad_id = tokenizer.pad_token_id
    collate_cls = make_collate_cls(pad_id)

    out_dir = C.OUTPUT_DIR / "predictions"

    # ST1
    det_dev = load_detection_records("dev", tokenizer)
    preds1 = _predict_cls(model, 0, det_dev, collate_cls, device)
    _write_jsonl(
        out_dir / "st1_predictions.jsonl",
        [{"id": r["index"], "label": p, "tag": C.TAG} for r, p in zip(det_dev, preds1)],
    )

    # ST2
    ext_dev = load_extraction_records("dev", tokenizer)
    spans2 = _predict_st2(model, ext_dev, device)
    _write_jsonl(
        out_dir / "st2_predictions.jsonl",
        [{"id": r["index"], "spans": s, "tag": C.TAG} for r, s in zip(ext_dev, spans2)],
    )

    # ST3
    idn_dev = load_identification_records("dev", tokenizer)
    preds3 = _predict_cls(model, 2, idn_dev, collate_cls, device)
    _write_jsonl(
        out_dir / "st3_predictions.jsonl",
        [{"id": r["index"], "label": p, "tag": C.TAG} for r, p in zip(idn_dev, preds3)],
    )

    print(f"predictions written to {out_dir}")


if __name__ == "__main__":
    main()
