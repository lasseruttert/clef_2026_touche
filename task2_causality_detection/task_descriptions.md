## Task: Causality Extraction
- Sub-Task 1: Given a natural language text, classify whether it contains causal information or not.
- Sub-Task 2: Given a natural language text, identify text spans that are good candidates to express events or concepts that are stated to partake in a causal relationship.
- Sub-Task 3: Given a natural language text and a candidate pair of events or concepts, E0 and E1, classify the type of causal relationship expressed between E0 and E1.

Given a natural language sentence, the extraction of causal statements, causality extraction, can be split into three steps; each of which is its own sub-task:

Classify for the entire sentence if contains causal information or not.
Identify text-spans that are good candidates to partake in a claimed or refuted causal relationship.
Given a pair of candidates, classify whether the sentence supports a causal relationship, refutes it (countercausal), or makes no statement how the pair relates (uncausal).

## Data Format
The following Python code opens the dataset and prints the first entry in the training set for each subtask (causality detection, causal candidate extraction, and causality identification) to the console.

from pathlib import Path
from datasets import load_dataset

path_to_dataset = Path(".")  # Replace me with the actual path!

dataset = load_dataset(str(path_to_dataset.absolute()), "causality detection")
print("Causality Detection")
print("Splits:", list(dataset.keys()))
print("Example:", dataset["train"][0])
print()

dataset = load_dataset(str(path_to_dataset.absolute()), "causal candidate extraction")
print("Causal Candidate Extraction")
print("Splits:", list(dataset.keys()))
print("Example:", dataset["train"][0])
print()

dataset = load_dataset(str(path_to_dataset.absolute()), "causality identification")
print("Causality Identification")
print("Splits:", list(dataset.keys()))
print("Example:", dataset["train"][0])
print()

## Submissions

Submission for Sub-Task 1
The submissions for Sub-Task 1 need to be made as a code submission.

The output of the code submission needs to be a JSONL file. Each line in the JSONL file should be in the following JSON format:

- id: The ID of the text that was classified.
- label: The label assigned by your classifier. 1 if the response contains (pro-/con-)causal information and 0 otherwise.
- tag: A tag that identifies your group and the method you used to produce the run.

Submission for Sub-Task 2
The submissions for Sub-Task 2 need to be made as a code submission.

The output of the code submission needs to be a JSONL file. Each line in the JSONL file should be in the following JSON format:

- id: The ID of the text that was classified.
- spans: The list of spans your submission predicted to partake in a causal relationship according to the input text. A span is a pair of two positive integers that give the start and end index in characrers.
- tag: A tag that identifies your group and the method you used to produce the run.

Submission for Sub-Task 3
The submissions for Sub-Task 3 need to be made as a code submission.

The output of the code submission needs to be a JSONL file. Each line in the JSONL file should be in the following JSON format:

- id: The ID of the text that was classified.
- label: The label assigned by your classifier. 0 if the marked spans (e0 and e1) are uncausal (nothing can be said about how e0 causally influences e1), 1 if they are causal (e0 causes e1) and 2 if they are countercausal (e0 does not cause e1).
- tag: A tag that identifies your group and the method you used to produce the run.

## Evaluation

Evaluation for Sub-Task 1
Sub-Task 1 is evaluated as a binary classification task using the F1-score.

Evaluation for Sub-Task 2
Sub-Task 2 is evaluated using F1-score.

Evaluation for Sub-Task 3
Sub-Task 3 is evaluated as a ternary classification task using the F1-score.