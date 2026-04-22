import itertools
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from transformers import DebertaV2Tokenizer, get_linear_schedule_with_warmup, set_seed
from torch.optim import AdamW

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from task2_causality_detection.deberta_joint import config as C
from task2_causality_detection.deberta_joint.model import JointDeBERTa
from task2_causality_detection.deberta_joint.data import (
    bio_to_spans, load_detection_records, load_extraction_records,
    load_identification_records, make_collate_bio, make_collate_cls, make_loader, span_f1,
)
from task2_causality_detection.common.pretty import log_best_result, quiet_transformers


def _setup_logging(run_dir: Path, tag: str) -> logging.Logger:
    run_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    logger = logging.getLogger("deberta_joint")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)
        fh = logging.FileHandler(run_dir / f"{tag}.log")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def _eval_cls(model, task_id: int, loader, device):
    all_logits, all_labels = [], []
    for batch in loader:
        labels = batch["labels"].to(device)
        inp = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        out = model(task_id=task_id, **inp)
        all_logits.append(out.logits.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    logits = np.concatenate(all_logits)
    preds = np.argmax(logits, axis=-1)
    labels_arr = np.array(all_labels)

    if task_id == 0:
        return {
            "f1_binary": float(f1_score(labels_arr, preds, pos_label=1)),
            "f1_macro_st1": float(f1_score(labels_arr, preds, average="macro")),
        }
    else:
        f1_per = f1_score(labels_arr, preds, average=None, labels=[0, 1, 2])
        return {
            "f1_macro_st3": float(f1_score(labels_arr, preds, average="macro")),
            "f1_uncausal": float(f1_per[0]),
            "f1_causal": float(f1_per[1]),
            "f1_countercausal": float(f1_per[2]),
        }


def _eval_st2(model, ext_dev_records, device, pad_id: int, batch_size: int = 32) -> float:
    f1s = []
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
            pred_ids = logits[j, :n].argmax(-1).tolist()
            pred_ids = [
                -100 if rec["labels"][k] == -100 else pred_ids[k]
                for k in range(n)
            ]
            pred_spans = bio_to_spans(rec["offset_mapping"], pred_ids)
            _, _, f1 = span_f1(pred_spans, rec["entity"])
            f1s.append(f1)
    return float(np.mean(f1s))


def eval_all(model, dev_loaders, ext_dev_records, device, pad_id: int) -> dict:
    model.eval()
    metrics = {}
    with torch.no_grad():
        metrics.update(_eval_cls(model, 0, dev_loaders[0], device))
        metrics["span_f1"] = _eval_st2(model, ext_dev_records, device, pad_id)
        metrics.update(_eval_cls(model, 2, dev_loaders[2], device))
    model.train()
    return metrics


def train_one(
    seed: int,
    output_dir: Path,
    logger: logging.Logger,
    tokenizer,
    det_train, ext_train, idn_train,
    det_dev, ext_dev, idn_dev,
    device,
) -> dict:
    set_seed(seed)
    pad_id = tokenizer.pad_token_id
    collate_cls = make_collate_cls(pad_id)
    collate_bio = make_collate_bio(pad_id)

    train_loaders = [
        make_loader(det_train, collate_cls, C.BATCH_SIZE, shuffle=True),
        make_loader(ext_train, collate_bio, C.BATCH_SIZE, shuffle=True),
        make_loader(idn_train, collate_cls, C.BATCH_SIZE, shuffle=True),
    ]
    dev_loaders = [
        make_loader(det_dev, collate_cls, C.BATCH_SIZE * 2, shuffle=False),
        None,
        make_loader(idn_dev, collate_cls, C.BATCH_SIZE * 2, shuffle=False),
    ]

    model = JointDeBERTa(C.MODEL_NAME)
    model.resize_token_embeddings(len(tokenizer))
    model = model.to(device)

    steps_per_epoch = max(len(tl) for tl in train_loaders)
    total_steps = steps_per_epoch * 3 * C.EPOCHS
    warmup_steps = int(total_steps * C.WARMUP_RATIO)
    logger.info(f"steps_per_epoch={steps_per_epoch}  total_steps={total_steps}  warmup={warmup_steps}")

    optimizer = AdamW(model.parameters(), lr=C.LR, weight_decay=C.WEIGHT_DECAY)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best_combined = -1.0
    best_state = None

    for epoch in range(C.EPOCHS):
        model.train()
        cyc = [itertools.cycle(tl) for tl in train_loaders]
        task_loss = [0.0, 0.0, 0.0]

        for _ in range(steps_per_epoch):
            for t in range(3):
                batch = next(cyc[t])
                batch = {k: v.to(device) for k, v in batch.items()}
                out = model(task_id=t, **batch)
                out.loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                task_loss[t] += out.loss.item()

        logger.info(
            f"[train] epoch {epoch + 1}/{C.EPOCHS}  "
            f"loss_det={task_loss[0] / steps_per_epoch:.4f}  "
            f"loss_ext={task_loss[1] / steps_per_epoch:.4f}  "
            f"loss_idn={task_loss[2] / steps_per_epoch:.4f}"
        )

        metrics = eval_all(model, dev_loaders, ext_dev, device, pad_id)
        logger.info(
            f"[eval ] epoch {epoch + 1}/{C.EPOCHS}  "
            + "  ".join(f"{k}={v:.4f}" for k, v in sorted(metrics.items()))
        )

        combined = (metrics["f1_binary"] + metrics["span_f1"] + metrics["f1_macro_st3"]) / 3
        if combined > best_combined:
            best_combined = combined
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            logger.info(f"  -> new best  combined={combined:.4f}")

    best_dir = output_dir / "best"
    best_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": best_state, "vocab_size": len(tokenizer)}, best_dir / "model.pt")
    tokenizer.save_pretrained(str(best_dir))

    model.load_state_dict(best_state)
    model.to(device)
    return eval_all(model, dev_loaders, ext_dev, device, pad_id)


def main():
    quiet_transformers()
    logger = _setup_logging(C.OUTPUT_DIR, C.TAG)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"device={device}")

    tokenizer = DebertaV2Tokenizer.from_pretrained(C.MODEL_NAME)
    tokenizer.add_special_tokens({"additional_special_tokens": C.SPECIAL_TOKENS})

    logger.info("loading data")
    det_train = load_detection_records("train", tokenizer)
    ext_train = load_extraction_records("train", tokenizer)
    idn_train = load_identification_records("train", tokenizer)
    det_dev = load_detection_records("dev", tokenizer)
    ext_dev = load_extraction_records("dev", tokenizer)
    idn_dev = load_identification_records("dev", tokenizer)
    logger.info(
        f"  detection  train={len(det_train)} dev={len(det_dev)}"
        f"  extraction train={len(ext_train)} dev={len(ext_dev)}"
        f"  identification train={len(idn_train)} dev={len(idn_dev)}"
    )

    final_metrics = train_one(
        C.SEED, C.OUTPUT_DIR, logger,
        tokenizer,
        det_train, ext_train, idn_train,
        det_dev, ext_dev, idn_dev,
        device,
    )
    log_best_result(C.RESULTS_LOG_ST1, C.TAG, {
        "f1_binary": final_metrics["f1_binary"],
        "f1_macro": final_metrics["f1_macro_st1"],
    })
    log_best_result(C.RESULTS_LOG_ST2, C.TAG, {
        "span_f1": final_metrics["span_f1"],
    })
    log_best_result(C.RESULTS_LOG_ST3, C.TAG, {
        "f1_macro": final_metrics["f1_macro_st3"],
        "f1_uncausal": final_metrics["f1_uncausal"],
        "f1_causal": final_metrics["f1_causal"],
        "f1_countercausal": final_metrics["f1_countercausal"],
    })
    logger.info("done  " + "  ".join(f"{k}={v:.4f}" for k, v in sorted(final_metrics.items())))


if __name__ == "__main__":
    main()
