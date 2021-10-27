import torch
from torch import nn
from torch.nn import Softmax

from src.transformer.modules import Dense, TransformerEncoder


class TransformerMeanClassifier(nn.Module):

    def __init__(self, config, num_classes, vocab_size):
        super().__init__()
        conf_trans = config['transformer']
        self.transformer_encoder = TransformerEncoder(num_layers=conf_trans['num_layers'],
                                                      num_heads=conf_trans['num_heads'],
                                                      d_model=conf_trans['dim_model'], d_ff=conf_trans['dim_ff'],
                                                      input_vocab_size=vocab_size,
                                                      max_position_encoding=conf_trans['max_position_encoding'])
        self.dense = Dense(input_dim=conf_trans['dim_model'], hidden_dim=num_classes)
        self.softmax = Softmax()

    def forward(self, x, mask):
        transformer_out = self.transformer_encoder(x, mask)
        transformer_out_mean = torch.mean(transformer_out, axis=-2)  # [batch_size, d_model]
        out = self.dense(transformer_out_mean)
        preds = self.softmax(out)
        return preds



class TransformerBERTClassifier(nn.Module):

    def __init__(self, config):
        super().__init__()
        conf_trans = config['transformer']
        num_classes = config['data']['num_classes']
        vocab_size = config['transformer']['vocab_size']
        self.encoder = TransformerEncoder(num_layers=conf_trans['num_layers'],
                                          num_heads=conf_trans['num_heads'],
                                          d_model=conf_trans['dim_model'], d_ff=conf_trans['dim_ff'],
                                          input_vocab_size=vocab_size,
                                          max_position_encoding=conf_trans['max_position_encoding'])
        self.dense = Dense(input_dim=conf_trans['dim_model'], hidden_dim=num_classes)
        self.softmax = Softmax()

    def forward(self, x, mask):
        transformer_out = self.encoder(x, mask)
        transformer_out_mean = torch.mean(transformer_out, axis=-2)  # [batch_size, d_model]
        out = self.dense(transformer_out_mean)
        preds = self.softmax(out)
        return preds
