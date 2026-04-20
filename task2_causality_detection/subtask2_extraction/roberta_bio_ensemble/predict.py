import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from task2_causality_detection.common.data_paths import FILES
from task2_causality_detection.subtask2_extraction.roberta_bio_ensemble import config as C
from task2_causality_detection.subtask2_extraction.roberta_bio_baseline.bio import bio_to_spans


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="dev", choices=["train", "dev"])
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--tag", default=C.TAG)
    p.add_argument("--runs_dir", default=str(C.OUTPUT_DIR))
    args = p.parse_args()

    runs_dir = Path(args.runs_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(str(runs_dir / f"seed_{C.SEEDS[0]}" / "best"),
                                        add_prefix_space=True)
    models = []
    for seed in C.SEEDS:
        m = AutoModelForTokenClassification.from_pretrained(str(runs_dir / f"seed_{seed}" / "best"))
        m.eval().to(device)
        models.append(m)

    df = pd.read_json(FILES["extraction"][args.split], lines=True)

    with args.out.open("w", encoding="utf-8") as f:
        for i in range(0, len(df), 32):
            batch = df.iloc[i:i + 32]
            enc = tok(batch["text"].tolist(), truncation=True, max_length=C.MAX_LENGTH,
                      padding=True, return_offsets_mapping=True,
                      return_special_tokens_mask=True, return_tensors="pt")
            offsets = enc.pop("offset_mapping").tolist()
            special = enc.pop("special_tokens_mask").tolist()
            enc_device = {k: v.to(device) for k, v in enc.items()}
            with torch.no_grad():
                avg_logits = sum(m(**enc_device).logits for m in models) / len(models)
            preds = avg_logits.argmax(-1).cpu().tolist()
            for j, idx in enumerate(batch["index"].tolist()):
                pred_ids = [-100 if special[j][k] else preds[j][k] for k in range(len(offsets[j]))]
                spans = bio_to_spans(offsets[j], pred_ids)
                f.write(json.dumps({"id": idx, "spans": [list(s) for s in spans],
                                    "tag": args.tag}) + "\n")


if __name__ == "__main__":
    main()
