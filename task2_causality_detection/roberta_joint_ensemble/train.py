import logging
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, set_seed

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from task2_causality_detection.roberta_joint_ensemble import config as C
from task2_causality_detection.roberta_joint.model import JointRoBERTa
from task2_causality_detection.roberta_joint.train import train_one
from task2_causality_detection.roberta_joint.data import (
    bio_to_spans, load_detection_records, load_extraction_records,
    load_identification_records, make_collate_cls, make_loader, span_f1,
)
from task2_causality_detection.common.pretty import log_best_result, quiet_transformers


def _setup_logging(run_dir: Path, tag: str) -> logging.Logger:
    run_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    logger = logging.getLogger("roberta_joint_ensemble")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)
        fh = logging.FileHandler(run_dir / f"{tag}.log")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


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


def _collect_st2_logits(model, ext_dev_records, device, batch_size=32):
    """Returns list of (n_tokens, 3) arrays, one per sample."""
    pad_id = 1
    per_sample = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(ext_dev_records), batch_size):
            batch_recs = ext_dev_records[i : i + batch_size]
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


def main():
    quiet_transformers()
    logger = _setup_logging(C.OUTPUT_DIR, C.TAG)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"device={device}")

    tokenizer = AutoTokenizer.from_pretrained(C.MODEL_NAME, add_prefix_space=True)
    tokenizer.add_special_tokens({"additional_special_tokens": C.SPECIAL_TOKENS})

    logger.info("loading data")
    det_train = load_detection_records("train", tokenizer)
    ext_train = load_extraction_records("train", tokenizer)
    idn_train = load_identification_records("train", tokenizer)
    det_dev = load_detection_records("dev", tokenizer)
    ext_dev = load_extraction_records("dev", tokenizer)
    idn_dev = load_identification_records("dev", tokenizer)

    # Phase 1: train each seed
    for seed in C.SEEDS:
        logger.info(f"--- seed {seed} ---")
        train_one(
            seed, C.OUTPUT_DIR / f"seed_{seed}", logger,
            tokenizer,
            det_train, ext_train, idn_train,
            det_dev, ext_dev, idn_dev,
            device,
        )

    # Phase 2: ensemble eval — collect logits from all seeds
    logger.info("--- ensemble eval ---")
    collate_cls = make_collate_cls(tokenizer.pad_token_id)
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

    # ST1
    avg_st1 = np.mean(st1_logits_all, axis=0)
    preds_st1 = np.argmax(avg_st1, axis=-1)
    st1_labels = np.array([r["labels"] for r in det_dev])
    st1_metrics = {
        "f1_binary": float(f1_score(st1_labels, preds_st1, pos_label=1)),
        "f1_macro": float(f1_score(st1_labels, preds_st1, average="macro")),
    }

    # ST2
    f1s = []
    for i, rec in enumerate(ext_dev):
        avg_logits_i = np.mean([st2_logits_all[s][i] for s in range(len(C.SEEDS))], axis=0)
        pred_ids = avg_logits_i.argmax(-1).tolist()
        pred_ids = [
            -100 if rec["labels"][k] == -100 else pred_ids[k]
            for k in range(len(pred_ids))
        ]
        pred_spans = bio_to_spans(rec["offset_mapping"], pred_ids)
        _, _, f1 = span_f1(pred_spans, rec["entity"])
        f1s.append(f1)
    span_f1_score = float(np.mean(f1s))

    # ST3
    avg_st3 = np.mean(st3_logits_all, axis=0)
    preds_st3 = np.argmax(avg_st3, axis=-1)
    st3_labels = np.array([r["labels"] for r in idn_dev])
    f1_per = f1_score(st3_labels, preds_st3, average=None, labels=[0, 1, 2])
    st3_metrics = {
        "f1_macro": float(f1_score(st3_labels, preds_st3, average="macro")),
        "f1_uncausal": float(f1_per[0]),
        "f1_causal": float(f1_per[1]),
        "f1_countercausal": float(f1_per[2]),
    }

    logger.info("[ensemble] st1  " + "  ".join(f"{k}={v:.4f}" for k, v in st1_metrics.items()))
    logger.info(f"[ensemble] st2  span_f1={span_f1_score:.4f}")
    logger.info("[ensemble] st3  " + "  ".join(f"{k}={v:.4f}" for k, v in st3_metrics.items()))

    log_best_result(C.RESULTS_LOG_ST1, C.TAG, st1_metrics)
    log_best_result(C.RESULTS_LOG_ST2, C.TAG, {"span_f1": span_f1_score})
    log_best_result(C.RESULTS_LOG_ST3, C.TAG, st3_metrics)


if __name__ == "__main__":
    main()
