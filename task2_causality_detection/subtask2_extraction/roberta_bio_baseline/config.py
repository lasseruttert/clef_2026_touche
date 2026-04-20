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
TAG = "roberta-bio-baseline"
RESULTS_LOG = Path(__file__).parents[1] / "results.log"

LABELS = ["O", "B-ENT", "I-ENT"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for i, l in enumerate(LABELS)}
