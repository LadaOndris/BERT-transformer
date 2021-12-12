import json

import torch

from src.data.loader import DataLoaderPreprocessor
from src.transformer.bert import BertClassifier


def create_model(config) -> BertClassifier:
    """
    Create a BERT model and load weights from Huggingface.
    Freezes all copied layers.
    :param config: Configuration specifying the architecture of the BERT model.
    :return: BertClassifer
    """
    bert_model = get_bert_model()
    classifier = BertClassifier(config)
    load_weights_from_bert(classifier, bert_model)
    freeze_encoder_layers(classifier)
    return classifier


def get_bert_model():
    """
    Load uncased HuggingFace model.
    """
    bert_model = torch.hub.load('huggingface/pytorch-transformers',
                                'model',
                                'bert-base-uncased')
    return bert_model


def get_bert_tokenizer():
    """
    Load uncased HuggingFace tokenizer.
    """
    bert_tokenizer = torch.hub.load('huggingface/pytorch-transformers',
                                    'tokenizer',
                                    'bert-base-uncased')
    return bert_tokenizer


def load_weights_from_bert(target_model: BertClassifier, source_model) -> None:
    """
    Loads weights from Huggingface's BERT model.
    """
    target_model.bert.embeddings.load_state_dict(source_model.embeddings.state_dict(), strict=False)
    target_model.bert.encoder.layers.load_state_dict(source_model.encoder.layer.state_dict())
    target_model.bert.pooler.load_state_dict(source_model.pooler.state_dict())


def freeze_encoder_layers(model: BertClassifier) -> None:
    """
    Freezes embedding and encoder layers.
    """
    for param in model.bert.embeddings.parameters():
        param.requires_grad = False
    for param in model.bert.encoder.parameters():
        param.requires_grad = False


def test_forward_pass(model: torch.nn.Module, data_loader):
    labels, input_ids, token_type_ids, masks = next(iter(data_loader))
    out = model(input_ids, token_type_ids)


if __name__ == '__main__':
    with open('src/config.json', 'r') as config_file:
        config = json.load(config_file)

    bert_tokenizer = get_bert_tokenizer()
    # out = bert_tokenizer('The computer age is just beginning.')
    # print(out)

    data_preprocessor = DataLoaderPreprocessor(batch_size=8,
                                               shuffle=True,
                                               tokenizer=bert_tokenizer)
    transformer_classifier = create_model(config)

    train_dataloader = data_preprocessor.get_train_data_loader()
    test_forward_pass(transformer_classifier, train_dataloader)
