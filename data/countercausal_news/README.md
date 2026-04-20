---
license: cc-by-4.0
task_categories:
- text-classification
- token-classification
language:
- en
multilinguality:
- monolingual
size_categories:
- 1K<n<10K
tags:
- causality
pretty_name: Countercausal News Corpus
---

# The Dataset
This folder contains the *Countercausal News Corpus* dataset files. Each file name has the form `<task>-<split>.jsonl`, where `<task>` is one of the three tasks: `causality-detection`, `causal-candidate-extraction`, and `causality-identification` and `<split>` is one of the two available splits: `train` or `dev`.

## Format
### Causality detection
Each line of the file contains a JSON object which has the following fields:
- `index`: A unique identifier for the entry.
- `label`: An integer that is `0` if the text is uncausal and `1` otherwise.
- `text`: A string that should be classified.

**Example:**
```json
{"index":"ccnc_train_10_150_2094_0","label":0,"text":"The union also holds the Shahjahanpur toll plaza ."}
```

### Causal candidate extraction
Each line of the file contains a JSON object which has the following fields:
- `index`: A unique identifier for the entry.
- `text`: A string containing the text in which entity spans should be marked.
- `entity`: A list of pairs of integer: the first integer marks the index in `text` at which the entity span starts and the second integer marks where it ends.

**Example:**
```json
{"index":"ccnc_train_10_161_3116_0","text":"`` There were demonstrations outside , but the meeting of the PEC continued , '' he said .","entity":[[14,36],[43,75]]}
```

### Causality identification
Each line of the file contains a JSON object which has the following fields:
- `index`: A unique identifier for the entry.
- `text`: A string with marked entity spans (`<e0>...</e0>` and `<e1>...</e1>`).
- `label`: An integer indicating the type of the relationship: `0` for uncausal, `1` for causal and `2` for countercausal.

**Example:**
```json
{"index":"ccnc_train_10_139_1388_0_0","text":"The company said <e1>negotiations were continuing between management and NUM officials<\/e1> with in a bid <e0>to bring an end to the strike<\/e0> .","label":1}
```

## Loading the Dataset
### Using HF Datasets
The Countercausal News Corpus is stored to be Hugging Face Datasets compatible. This means that you can simply load it using

```py
from datasets import load_dataset

dset = load_dataset("<path-to-dataset>", "<task>")
```

from inside the repo root, where `"<task>"` is one of the supported tasks: `"causality detection"`, `"causal candidate extraction"`, and `"causality identification"`.

### Using Pandas
Altneratively you can load the jsonl files directly through
```py
import pandas as pd

df = pd.read_json("<path-to-dataset>/<task>-<split>.jsonl", lines=True)
```
from inside the repo root, where `<task>` is one of the supported tasks: `"causality-detection"`, `"causal-candidate-extraction"`, and `"causality-identification"`, and `<split>` is one of `train` or `dev`.
