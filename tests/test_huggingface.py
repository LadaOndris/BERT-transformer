import json
from unittest import TestCase

import torch

from src.transformer.huggingface import get_bert_model, get_bert_tokenizer, load_weights_from_bert
from src.transformer.bert import BertClassifier


class Test(TestCase):

    def setUp(self):
        with open('src/config.json', 'r') as config_file:
            config = json.load(config_file)

        self.model = BertClassifier(config)
        self.huggingface_model = get_bert_model()
        self.huggingface_tokenizer = get_bert_tokenizer()

    def test_load_weights_from_bert(self):
        tokenized_text = self.huggingface_tokenizer('This is a sample sentence.')
        input_ids, type_ids = tokenized_text['input_ids'], tokenized_text['token_type_ids']
        input_ids = torch.tensor([input_ids])
        type_ids = torch.tensor([type_ids])

        load_weights_from_bert(self.model, self.huggingface_model)

        self.model.eval()
        self.huggingface_model.eval()

        huggingface_output = self.huggingface_model(input_ids, type_ids)
        pooler_output = huggingface_output.pooler_output
        output = self.model.bert(input_ids, type_ids)

        torch.testing.assert_allclose(output, pooler_output, atol=1e-4, rtol=1e-4)
