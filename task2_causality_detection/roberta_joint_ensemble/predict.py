import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from task2_causality_detection.roberta_joint_ensemble import config as C
from task2_causality_detection.roberta_joint.model import JointRoBERTa
from task2_causality_detection.roberta_joint.data import (
    bio_to_spans, load_detection_records, load_extraction_records,
    load_identification_records, make_collate_cls, make_loader,
)
from task2_causality_detection.common.pretty import quiet_transformers


def _collect_cls_logits(model, task_id, records, collate_cls, batch_size, device):
    loader = make_loader(records, collate_cls, batch_size, shuffle=False)
    all_logits = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            inp = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            out = model(task_id=task_id, **inp)
            all_logits.append(out.logits.cpu().numpy())
    return np.concatenate(all_logits)


def _collect_st2_logits(model, ext_records, device, batch_size=32):
    pad_id = 1
    per_sample = []
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
                per_sample.append(logits[j, :n])
    return per_sample


def _write_jsonl(path: Path, rows: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def main():
    quiet_transformers()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(
        str(C.OUTPUT_DIR / f"seed_{C.SEEDS[0]}" / "best")
    )
    collate_cls = make_collate_cls(tokenizer.pad_token_id)

    det_dev = load_detection_records("dev", tokenizer)
    ext_dev = load_extraction_records("dev", tokenizer)
    idn_dev = load_identification_records("dev", tokenizer)

    st1_logits_all, st2_logits_all, st3_logits_all = [], [], []
    for seed in C.SEEDS:
        ckpt = torch.load(
            C.OUTPUT_DIR / f"seed_{seed}" / "best" / "model.pt", map_location="cpu"
        )
        model = JointRoBERTa(C.MODEL_NAME)
        model.resize_token_embeddings(ckpt["vocab_size"])
        model.load_state_dict(ckpt["model_state"])
        model = model.to(device)

        st1_logits_all.append(_collect_cls_logits(model, 0, det_dev, collate_cls, C.BATCH_SIZE * 2, device))
        st2_logits_all.append(_collect_st2_logits(model, ext_dev, device))
        st3_logits_all.append(_collect_cls_logits(model, 2, idn_dev, collate_cls, C.BATCH_SIZE * 2, device))

    out_dir = C.OUTPUT_DIR / "predictions"

    # ST1
    preds1 = np.argmax(np.mean(st1_logits_all, axis=0), axis=-1).tolist()
    _write_jsonl(
        out_dir / "st1_predictions.jsonl",
        [{"id": r["index"], "label": p, "tag": C.TAG} for r, p in zip(det_dev, preds1)],
    )

    # ST2
    spans2 = []
    for i, rec in enumerate(ext_dev):
        avg_logits_i = np.mean([st2_logits_all[s][i] for s in range(len(C.SEEDS))], axis=0)
        pred_ids = avg_logits_i.argmax(-1).tolist()
        pred_ids = [
            -100 if rec["labels"][k] == -100 else pred_ids[k]
            for k in range(len(pred_ids))
        ]
        spans = bio_to_spans(rec["offset_mapping"], pred_ids)
        spans2.append([[s, e] for s, e in spans])
    _write_jsonl(
        out_dir / "st2_predictions.jsonl",
        [{"id": r["index"], "spans": s, "tag": C.TAG} for r, s in zip(ext_dev, spans2)],
    )

    # ST3
    preds3 = np.argmax(np.mean(st3_logits_all, axis=0), axis=-1).tolist()
    _write_jsonl(
        out_dir / "st3_predictions.jsonl",
        [{"id": r["index"], "label": p, "tag": C.TAG} for r, p in zip(idn_dev, preds3)],
    )

    print(f"predictions written to {out_dir}")


if __name__ == "__main__":
    main()
