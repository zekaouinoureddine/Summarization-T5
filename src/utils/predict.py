import config
from model import CNNDailyMailModel
from process import Process

import torch


model = CNNDailyMailModel()
model.to(config.DEVICE)
model.load_state_dict(
    torch.load(
        config.MODEL_PATH,
        map_location=torch.device(config.DEVICE)
    )
)


def predict(article_text):
    data = Process(article_text)
    input_ids, attention_mask = data.pre_process()
    input_ids = input_ids.to(config.DEVICE)
    attention_mask = attention_mask.to(config.DEVICE)

    with torch.no_grad():
        generated_ids = model.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=150,
            num_beams=2,
            repetition_penalty=2.5,
            length_penalty=1.0,
            early_stopping=True
        )

    predicted_high = data.post_process(generated_ids)
    return predicted_high
