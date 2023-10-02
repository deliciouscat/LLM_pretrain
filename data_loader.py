from torch.utils.data import Dataset

# 데이터셋 클래스 정의
class BuildDataset(Dataset):
    def __init__(self, text_list, tokenizer, max_length):
        self.text_list = text_list
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.text_list)

    def __getitem__(self, idx):
        inputs = self.tokenizer(
            self.text_list[idx],
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=self.max_length
        )
        return inputs