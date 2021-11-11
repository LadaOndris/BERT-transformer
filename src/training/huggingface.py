import json

import torch

from src.training.data_loader import DataLoaderPreprocessor
from src.transformer.bert import BertClassifier


def create_model(config) -> BertClassifier:
    bert_model = get_bert_model()
    classifier = BertClassifier(config)
    load_weights_from_bert(classifier, bert_model)
    return classifier


def load_weights_from_bert(target_model: BertClassifier, source_model):
    target_model.bert.embeddings.load_state_dict(source_model.embeddings.state_dict(), strict=False)
    target_model.bert.encoder.layers.load_state_dict(source_model.encoder.layer.state_dict())
    target_model.bert.pooler.load_state_dict(source_model.pooler.state_dict())


def get_bert_model():
    bert_model = torch.hub.load('huggingface/pytorch-transformers',
                                'model',
                                'bert-base-uncased')
    return bert_model


def get_bert_tokenizer():
    bert_tokenizer = torch.hub.load('huggingface/pytorch-transformers',
                                    'tokenizer',
                                    'bert-base-uncased')
    return bert_tokenizer


def test_forward_pass(model: torch.nn.Module, data_loader):
    for batch in data_loader:
        labels, input_ids, token_type_ids, masks = batch
        out = model(input_ids, token_type_ids)
        break


if __name__ == '__main__':
    with open('./src/config.json', 'r') as config_file:
        config = json.load(config_file)

    data_preprocessor = DataLoaderPreprocessor(batch_size=config['train']['batch_size'],
                                               shuffle=True,
                                               tokenizer=get_bert_tokenizer())
    transformer_classifier = create_model(config)

    train_dataloader = data_preprocessor.get_train_data_loader()
    test_forward_pass(transformer_classifier, train_dataloader)
