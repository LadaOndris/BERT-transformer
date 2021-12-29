import json
import time

import numpy as np
import torch

from src.transformer.bert import Bert
from src.transformer.huggingface import get_bert_model


def get_tensor():
    batch_size = 1
    seq_len = 221
    # model_dim = 768
    array = np.random.randint(1000, 10000, batch_size * seq_len)
    array = array.reshape([batch_size, seq_len])
    tensor = torch.tensor(array)
    token_type_ids = torch.tensor(np.zeros(shape=(batch_size, seq_len), dtype=int))
    return tensor, token_type_ids


def time_model(model, reps=50):
    tensor, token_type_ids = get_tensor()
    model(tensor, token_type_ids)

    start = time.time()
    for i in range(reps):
        model(tensor, token_type_ids)
    end = time.time()
    inference_time_reps = end - start
    inference_per_batch = inference_time_reps / reps
    print(F"Inference time: {inference_per_batch:2f}s")


def time_huggingface_bert_model():
    model = get_bert_model()
    time_model(model)


def time_my_bert_model():
    with open('src/config.json', 'r') as config_file:
        config = json.load(config_file)
    model = Bert(config)
    time_model(model)


if __name__ == "__main__":
    time_huggingface_bert_model()
    time_my_bert_model()
