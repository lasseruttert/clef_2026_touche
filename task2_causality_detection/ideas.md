## Sub-Task 1: Given a natural language text, classify whether it contains causal information or not.


## Sub-Task 2: Given a natural language text, identify text spans that are good candidates to express events or concepts that are stated to partake in a causal relationship.
- use model from subtask 1 to check if there is even any causal information or not
- span-pointer head (QA-style start/end logits) instead of BIO tagging — BIO is the ceiling here

## Sub-Task 3: Given a natural language text and a candidate pair of events or concepts, E0 and E1, classify the type of causal relationship expressed between E0 and E1.
- let an llm create an explanation of the relationship and give that as input
- swap e0 and e1 for negative case 

## Thoughts

- Encoder Only Transformer trained on a single task for all three tasks (baseline) (ensemble (maybe across different nlp methods), k-fold)
- Encoder Only Transformer with 3 heads trained jointly on all three tasks at once
- LLM Prompting
- LoRA Finetuning with Next Token Prediction
- LLMs with special heads per task (repeat input for task 2)
- encoder-decoder
- continued pretraining?

- remove the \uXXXX things from data
- basic cleanup of random chars and so on
- check label distribution

- interesting features to also give as input?

## Results so far (2026-04-20)

| model                | ST1 f1_binary | ST2 span_f1 | ST3 f1_macro |
|----------------------|---------------|-------------|--------------|
| roberta-baseline     | 0.8687        | 0.5747      | 0.7079       |
| roberta-ensemble     | **0.8911**    | 0.5653      | 0.7174       |
| roberta-joint        | 0.8834        | **0.6078**  | **0.7549**   |
| roberta-joint-ens    | 0.8762        | 0.5988      | 0.7105       |

- Joint multi-task training clearly helps ST2 (+3.3%) and ST3 (+4.7%) — shared encoder benefits from cross-task signal
- Ensemble consistently hurts the joint model (ST3: 0.7549 → 0.7105) — seeds likely find incompatible solutions in the harder multi-task landscape; don't pursue joint ensemble further
- Per-task ensemble still best for ST1; joint model doesn't help ST1
- BIO tagging is still the bottleneck for ST2 even with joint training — architecture change needed
- Best current submission: roberta-ensemble for ST1, roberta-joint for ST2 and ST3