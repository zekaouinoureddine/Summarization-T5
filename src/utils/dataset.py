import config

import torch
from torch.utils.data import Dataset


class CNNDailyMailDataset(Dataset):
    def __init__(self, article, highlights):
        self.article = article
        self.highlights = highlights

        self.tokenizer = config.TOKENIZER
        self.art_max_len = config.ART_MAX_LEN
        self.hig_max_len = config.ART_MAX_LEN

    def __len__(self):
        return len(self.highlights)

    def __getitem__(self, index):
        article = str(self.article[index])
        highlights = str(self.highlights[index])

        article = " ".join(article.split())
        highlights = " ".join(highlights.split())

        article_inputs = self.tokenizer.batch_encode_plus(
            [article],
            max_length=self.art_max_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length"
        )
        highlights_inputs = self.tokenizer.batch_encode_plus(
            [highlights],
            max_length=self.hig_max_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length"
        )

        article_input_ids = article_inputs["input_ids"]
        article_attention_mask = article_inputs["attention_mask"]

        highlights_input_ids = highlights_inputs["input_ids"]
        highlights_attention_mask = highlights_inputs["attention_mask"]

        labels = [label if label != 0 else -
                  100 for label in highlights_input_ids]

        return {
            "article_ids": torch.tensor(highlights_input_ids, dtype=torch.long).squeeze(),
            "article_mask": torch.tensor(article_attention_mask, dtype=torch.long).squeeze(),
            "highlights_ids": torch.tensor(highlights_input_ids, dtype=torch.long).squeeze(),
            "highlights_mask": torch.tensor(highlights_attention_mask, dtype=torch.long).squeeze(),
            "labels": torch.tensor(labels, dtype=torch.long).squeeze(),
        }
