import config

from torch import nn


class CNNDailyMailModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = config.MODEL

    def forward(self, input_ids, attention_mask, decoder_attention_mask, labels=None):
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
        )
        return output.loss, output.logits
