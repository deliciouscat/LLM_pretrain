import pytorch_lightning as pl
from transformers import DebertaForMaskedLM, AdamW


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
        # ODE Solver
        # outputs = self(input_ids, attention_mask)
        # loss = outputs.loss
        outputs = self(input_ids, attention_mask)
        x  = outputs.hidden_states[0]
        y0 = outputs.hidden_states[-1]
        
        y1 = x  + 0.5*y0 + 0.5 * self.encoder(x  + y0)
        y2 = y0 + 0.5*y1 + 0.5 * self.encoder(y0 + y1)
        y3 = y1 + 0.5*y2 + 0.5 * self.encoder(y1 + y2)
        
        # 별도의 예측값 y3(==y_hat)을 사용한 loss를 어떻게 적용할지? normalizing 포함해서
        # output.loss = 
        loss = outputs.loss
        return {"loss": loss}

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr)
        return optimizer