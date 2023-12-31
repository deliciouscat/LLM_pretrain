from transformers import TrainingArguments, Trainer
from transformers import AutoTokenizer
from data_loader import BuildDataset
from model import MLMModel

text_list = ["This is an example sentence.", "Another sentence for training."]
tokenizer = AutoTokenizer.from_pretrained("snunlp/KR-Medium",
                                          do_lower_case=False)
max_length = 256  # 문장의 최대 길이

# 데이터셋 생성
dataset = BuildDataset(text_list, tokenizer, max_length)

# Lightning 모델 생성
mlm_model = MLMModel('microsoft/deberta-v3-base', tokenizer, max_length)

# 학습 설정
training_args = TrainingArguments(
    output_dir='./mlm_output',
    num_train_epochs=2,
    per_device_train_batch_size=2,
    save_steps=10,
    save_total_limit=2,
    logging_dir='./mlm_logs',
    evaluation_strategy='steps',
    eval_steps=10,
)

# Trainer 생성 및 학습
trainer = Trainer(
    model=mlm_model,
    args=training_args,
    data_collator=None,
    train_dataset=dataset,
)

trainer.train()






