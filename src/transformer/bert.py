import torch
from torch import nn
from torch.nn import Dropout, Tanh

from src.transformer.modules import Dense, Embedding, LayerNormalization, TransformerEncoder
from src.transformer.operations import create_padding_mask


class BertClassifier(nn.Module):

    def __init__(self, config):
        super(BertClassifier, self).__init__()
        d_model = config['transformer']['dim_model']
        n_classes = config['data']['num_classes']
        self.bert = Bert(config)
        self.dropout = Dropout(p=0.1)
        self.classifier = Dense(d_model, n_classes)

    def forward(self, input_ids, token_type_ids):
        # input_ids = [batch_size, seq_len]
        output_cls = self.bert(input_ids, token_type_ids)  # [batch_size, d_model]
        logits_cls = self.classifier(output_cls)  # [batch_size, n_classes]
        return logits_cls


class Bert(nn.Module):

    def __init__(self, config):
        super(Bert, self).__init__()
        conf_trans = config['transformer']
        self.embeddings = BertEmbeddings(conf_trans['vocab_size'],
                                         conf_trans['dim_model'],
                                         conf_trans['max_position_encoding'],
                                         layer_norm_eps=conf_trans['layer_norm_eps'])
        self.encoder = TransformerEncoder(num_layers=conf_trans['num_layers'], d_model=conf_trans['dim_model'],
                                          num_heads=conf_trans['num_heads'], d_ff=conf_trans['dim_ff'],
                                          layer_norm_eps=conf_trans['layer_norm_eps'])
        self.pooler = BertPooler(conf_trans['dim_model'])

    def forward(self, x, token_type_ids):
        mask = create_padding_mask(x)
        embeddings = self.embeddings(x, token_type_ids)
        output = self.encoder(embeddings, mask)  # [batch_size, seq_len, d_model]
        # Take the first value (at the position of the [CLS] token)
        output_cls = output[:, 0, :]  # [batch_size, d_model]
        pooled = self.pooler(output_cls)  # [batch_size, d_model]
        return pooled


class BertEmbeddings(nn.Module):

    def __init__(self, vocab_size, dim_embed, max_seq_len, layer_norm_eps):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = Embedding(vocab_size, dim_embed)
        self.position_embeddings = Embedding(max_seq_len, dim_embed)
        self.token_type_embeddings = Embedding(2, dim_embed)
        self.LayerNorm = LayerNormalization(dim_embed, epsilon=layer_norm_eps)
        self.dropout = Dropout(p=0.1)
        position_ids = torch.arange(max_seq_len, dtype=torch.long).unsqueeze(0)
        self.register_buffer('position_ids', position_ids)

    def forward(self, x, seg):
        # x = (batch_size, seq_len)
        seq_len = x.size(1)
        # pos = torch.arange(seq_len, dtype=torch.long)
        pos = self.position_ids[:, :seq_len]
        pos = pos.expand_as(x)  # (1, seq_len,) -> (batch_size, seq_len)
        embeddings = self.word_embeddings(x) + self.position_embeddings(pos) + self.token_type_embeddings(seg)
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertPooler(nn.Module):

    def __init__(self, dim_embed):
        super(BertPooler, self).__init__()
        self.dense = Dense(dim_embed, dim_embed)
        self.activation = Tanh()

    def forward(self, x):
        return self.activation(self.dense(x))  # [batch_size, dim_embed]
