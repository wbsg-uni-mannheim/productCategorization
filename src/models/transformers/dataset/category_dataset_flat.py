import torch

class CategoryDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, le_dict):
        #Preprocess encodings
        self.encodings = tokenizer(texts, padding=True, truncation=True)

        #Preprocess labels
        self.labels = [le_dict.get(x, len(le_dict) + 1) for x in labels]

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]).to(torch.int64) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx]).to(torch.int64)
        return item

    def __len__(self):
        return len(self.labels)

