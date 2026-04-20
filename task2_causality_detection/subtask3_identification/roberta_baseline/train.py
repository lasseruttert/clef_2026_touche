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
from task2_causality_detection.common.pretty import attach, quiet_transformers
from task2_causality_detection.subtask3_identification.roberta_baseline import config as C


def load_split(split: str) -> Dataset:
    df = pd.read_json(FILES["identification"][split], lines=True)
    # keep <e0>/<e1> markers verbatim; no offset-shifting cleanup
    return Dataset.from_pandas(df[["index", "text", "label"]], preserve_index=False)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    f1_per = f1_score(labels, preds, average=None, labels=[0, 1, 2])
    return {
        "f1_macro": f1_score(labels, preds, average="macro"),
        "f1_uncausal": f1_per[0],
        "f1_causal": f1_per[1],
        "f1_countercausal": f1_per[2],
    }


def main():
    quiet_transformers()
    set_seed(C.SEED)
    tokenizer = AutoTokenizer.from_pretrained(C.MODEL_NAME)
    tokenizer.add_special_tokens({"additional_special_tokens": C.SPECIAL_TOKENS})
    model = AutoModelForSequenceClassification.from_pretrained(C.MODEL_NAME, num_labels=C.NUM_LABELS)
    model.resize_token_embeddings(len(tokenizer))

    train_ds = load_split("train")
    dev_ds = load_split("dev")

    def tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=C.MAX_LENGTH)

    train_ds = train_ds.map(tok, batched=True)
    dev_ds = dev_ds.map(tok, batched=True)

    args = TrainingArguments(
        output_dir=str(C.OUTPUT_DIR),
        num_train_epochs=C.EPOCHS,
        per_device_train_batch_size=C.BATCH_SIZE,
        per_device_eval_batch_size=C.BATCH_SIZE * 2,
        learning_rate=C.LR,
        weight_decay=C.WEIGHT_DECAY,
        warmup_ratio=C.WARMUP_RATIO,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=1,
        seed=C.SEED,
        report_to="none",
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
    trainer.save_model(str(C.OUTPUT_DIR / "best"))


if __name__ == "__main__":
    main()
