import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from task2_causality_detection.common.data_paths import FILES
from task2_causality_detection.common.preprocessing import clean_loose
from task2_causality_detection.common.pretty import attach, log_best_result, quiet_transformers, setup_logging
from task2_causality_detection.subtask1_detection.roberta_ensemble import config as C


def load_split(split: str) -> Dataset:
    df = pd.read_json(FILES["detection"][split], lines=True)
    df["text"] = df["text"].map(clean_loose)
    return Dataset.from_pandas(df[["index", "text", "label"]], preserve_index=False)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "f1_binary": f1_score(labels, preds, pos_label=1),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }


def train_one(seed: int, train_ds: Dataset, dev_ds: Dataset, tokenizer):
    model = AutoModelForSequenceClassification.from_pretrained(C.MODEL_NAME, num_labels=2)
    seed_dir = C.OUTPUT_DIR / f"seed_{seed}"
    args = TrainingArguments(
        output_dir=str(seed_dir),
        num_train_epochs=C.EPOCHS,
        per_device_train_batch_size=C.BATCH_SIZE,
        per_device_eval_batch_size=C.BATCH_SIZE * 2,
        learning_rate=C.LR,
        weight_decay=C.WEIGHT_DECAY,
        warmup_ratio=C.WARMUP_RATIO,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_binary",
        greater_is_better=True,
        save_total_limit=1,
        seed=seed,
        report_to="tensorboard",
        logging_strategy="epoch",
        disable_tqdm=True,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )
    attach(trainer)
    trainer.train()
    trainer.save_model(str(seed_dir / "best"))


def main():
    quiet_transformers()
    setup_logging(C.OUTPUT_DIR, C.TAG)
    tokenizer = AutoTokenizer.from_pretrained(C.MODEL_NAME)

    train_ds = load_split("train")
    dev_ds = load_split("dev")

    def tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=C.MAX_LENGTH)

    train_ds = train_ds.map(tok, batched=True)
    dev_ds = dev_ds.map(tok, batched=True)

    log = logging.getLogger("task2_causality_detection.common.pretty")
    for seed in C.SEEDS:
        log.info(f"--- seed {seed} ---")
        set_seed(seed)
        train_one(seed, train_ds, dev_ds, tokenizer)

    log.info("--- ensemble eval ---")
    all_logits = []
    for seed in C.SEEDS:
        model = AutoModelForSequenceClassification.from_pretrained(
            str(C.OUTPUT_DIR / f"seed_{seed}" / "best")
        )
        pred_out = Trainer(
            model=model,
            args=TrainingArguments(
                output_dir=str(C.OUTPUT_DIR / "_tmp"),
                per_device_eval_batch_size=C.BATCH_SIZE * 2,
                report_to="none",
                disable_tqdm=True,
            ),
            processing_class=tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer),
        ).predict(dev_ds)
        all_logits.append(pred_out.predictions)

    avg_logits = np.mean(all_logits, axis=0)
    labels = np.array(dev_ds["label"])
    metrics = compute_metrics((avg_logits, labels))
    log.info("[ensemble] " + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
    log_best_result(C.RESULTS_LOG, C.TAG, metrics)


if __name__ == "__main__":
    main()
