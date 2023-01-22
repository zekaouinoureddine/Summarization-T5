import config

import torch



class Process:
    def __init__(self, article_text):
        self.article_text = article_text
        self.art_max_len = config.ART_MAX_LEN
        self.tokenizer = config.TOKENIZER

    def pre_process(self):
        article_text = str(self.article_text)
        article_text = " ".join(article_text.split())

        inputs = self.tokenizer.batch_encode_plus(
            [article_text],
            max_length=self.art_max_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length"
        )

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(attention_mask, dtype=torch.long)
        )

    def post_process(self, generated_ids):
        preds = [
            self.tokenizer.decode(
                generated_id, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            for generated_id in generated_ids
        ]
        return " ".join(preds)
