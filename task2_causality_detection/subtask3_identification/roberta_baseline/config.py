from pathlib import Path

MODEL_NAME = "roberta-base"
MAX_LENGTH = 256
BATCH_SIZE = 16
EPOCHS = 5
LR = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
SEED = 42
OUTPUT_DIR = Path(__file__).parent / "runs"
TAG = "roberta-baseline"

SPECIAL_TOKENS = ["<e0>", "</e0>", "<e1>", "</e1>"]
NUM_LABELS = 3  # 0 uncausal, 1 causal, 2 countercausal
