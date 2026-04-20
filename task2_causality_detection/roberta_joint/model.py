import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional
from transformers import RobertaModel


@dataclass
class JointOutput:
    loss: Optional[torch.Tensor]
    logits: torch.Tensor


class JointRoBERTa(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained(model_name)
        hidden = self.roberta.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.head_detect = nn.Linear(hidden, 2)
        self.head_extract = nn.Linear(hidden, 3)
        self.head_identify = nn.Linear(hidden, 3)

    def resize_token_embeddings(self, new_num_tokens: int):
        self.roberta.resize_token_embeddings(new_num_tokens)

    def forward(
        self,
        task_id: int,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> JointOutput:
        out = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        seq = self.dropout(out.last_hidden_state)
        cls = seq[:, 0]

        if task_id == 0:
            logits = self.head_detect(cls)
            loss = nn.CrossEntropyLoss()(logits, labels) if labels is not None else None
        elif task_id == 1:
            logits = self.head_extract(seq)
            if labels is not None:
                loss = nn.CrossEntropyLoss(ignore_index=-100)(
                    logits.view(-1, 3), labels.view(-1)
                )
            else:
                loss = None
        else:
            logits = self.head_identify(cls)
            loss = nn.CrossEntropyLoss()(logits, labels) if labels is not None else None

        return JointOutput(loss=loss, logits=logits)
