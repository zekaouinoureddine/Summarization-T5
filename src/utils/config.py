import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration


EPOCHS = 4
MAX_LEN = 512
HIG_MAX_LEN = 64
ART_MAX_LEN = 512
LEARNING_RATE = 3e-5
TRAIN_BATCH_SIZE = 4
VALID_BATCH_SIZE = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINT = "t5-base"
MODEL_PATH = "./models/output/t5ass.bin"

TOKENIZER = T5Tokenizer.from_pretrained(CHECKPOINT)
MODEL = T5ForConditionalGeneration.from_pretrained(CHECKPOINT, return_dict=True)