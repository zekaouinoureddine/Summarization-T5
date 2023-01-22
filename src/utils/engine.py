import config

import torch
from tqdm import tqdm


def train_fn(train_dataloader, model, optimizer, device):
    model.train()
    final_loss = 0
    fin_labels = []
    fin_outputs = []
    for data in tqdm(train_dataloader, total=len(train_dataloader)):
        art_ids = data["article_ids"].to(device)
        art_mask = data["article_mask"].to(device)

        labels = data["labels"].to(device)
        labels_attention_mask = data["highlights_mask"].to(device)

        loss, logits = model(
            input_ids=art_ids,
            attention_mask=art_mask,
            decoder_attention_mask=labels_attention_mask,
            labels=labels
        )
        final_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return final_loss/len(train_dataloader)


def eval_fn(valid_dataloader, model, device):
    model.eval()
    final_loss = 0
    decoded_labels = []
    decoded_preds = []
    with torch.no_grad():
        for data in tqdm(valid_dataloader, total=len(valid_dataloader)):
            art_ids = data["article_ids"].to(device)
            art_mask = data["article_mask"].to(device)

            labels = data["labels"].to(device)
            labels_attention_mask = data["highlights_mask"].to(device)

            loss, logits = model(
                input_ids=art_ids,
                attention_mask=art_mask,
                decoder_attention_mask=labels_attention_mask,
                labels=labels
            )
            final_loss += loss.item()

            generated_ids = model.model.generate(
                input_ids=art_ids,
                attention_mask=art_mask,
                max_length=150,
                num_beams=2,
                repetition_penalty=2.5,
                length_penalty=1.0,
                early_stopping=True
            )

            # Replace -100 in the labels as we can't decode them.
            labels = torch.where(labels != -100, labels, config.TOKENIZER.pad_token_id)

            decoded_preds.extend([config.TOKENIZER.decode(
                g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids])
            decoded_labels.extend([config.TOKENIZER.decode(
                t, skip_special_tokens=True, clean_up_tokenization_spaces=True)for t in labels])

    return decoded_preds, decoded_labels, final_loss/len(valid_dataloader)
