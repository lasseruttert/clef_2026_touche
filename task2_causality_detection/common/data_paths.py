from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "countercausal_news"

FILES = {
    "detection": {
        "train": DATA_DIR / "causality-detection-train.jsonl",
        "dev":   DATA_DIR / "causality-detection-dev.jsonl",
    },
    "extraction": {
        "train": DATA_DIR / "causal-candidate-extraction-train.jsonl",
        "dev":   DATA_DIR / "causal-candidate-extraction-dev.jsonl",
    },
    "identification": {
        "train": DATA_DIR / "causality-identification-train.jsonl",
        "dev":   DATA_DIR / "causality-identification-dev.jsonl",
    },
}
