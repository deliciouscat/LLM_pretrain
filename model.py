import torch
import pytorch_lightning as pl
from transformers import DebertaForMaskedLM, DebertaTokenizer, AdamW
from transformers import TrainingArguments, Trainer
import numpy as np


# PyTorch Lightning 모델 클래스 정의
class MLMModel(pl.LightningModule):
    def __init__(self, model_name, tokenizer, max_length, lr=2e-5):
        super().__init__()
        self.model = DebertaForMaskedLM.from_pretrained(model_name)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.lr = lr

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask=attention_mask)

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        outputs = self(input_ids, attention_mask)
        loss = outputs.loss
        return {"loss": loss}

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr)
        return optimizer
    
    