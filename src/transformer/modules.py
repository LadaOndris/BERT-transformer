import math

import torch
from torch import nn
from torch.nn import Dropout

from src.transformer.operations import softmax


class TransformerEncoder(nn.Module):

    def __init__(self, num_layers: int, d_model: int, num_heads: int, d_ff: int):
        super(TransformerEncoder, self).__init__()
        self.num_layers = num_layers

        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff)
                                     for _ in range(num_layers)])

    def forward(self, x, mask):
        for i in range(self.num_layers):
            x = self.layers[i](x, mask)
        return x


class Embedding(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(Embedding, self).__init__()
        self.output_dim = output_dim
        self.weight = nn.Parameter(torch.zeros([input_dim, output_dim]))
        torch.nn.init.xavier_normal_(self.weight)

    def forward(self, x):
        x_flattened = x.flatten()
        embeddings = torch.index_select(self.weight, dim=0, index=x_flattened)
        embeddings_out_shape = list(x.shape) + [self.output_dim]
        embeddings_reshaped = embeddings.reshape(embeddings_out_shape)
        return embeddings_reshaped


class EncoderLayer(nn.Module):

    def __init__(self, d_model: int, num_heads: int, d_ff: int):
        """

        :param d_model:
        :param num_heads: The number of (MHA, FFN) heads to use.
            Paper 'Attention is all you need' used 6 of them.
        :param d_ff: Number of hidden neurons in the point-wise feed forward network.
        """
        super(EncoderLayer, self).__init__()
        self.d_model = d_model

        self.attention = BertAttention(d_model, num_heads)
        self.intermediate = BertIntermediate(d_model, d_ff)
        self.output = BertOutput(d_ff, d_model)

    def forward(self, x, mask):
        attention = self.attention(x, x, x, mask)  # (batch_size, seq_len, d_model)
        intermediate = self.intermediate(attention)  # (batch_size, seq_len, d_model)
        output = self.output(intermediate, attention)
        return output


class BertAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super(BertAttention, self).__init__()

        self.self = BertSelfAttention(d_model, num_heads)
        self.output = BertOutput(d_model, d_model)

    def forward(self, query, key, value, mask):
        residual = query
        scaled_attention = self.self(query, key, value, mask)
        output =  self.output(scaled_attention, residual)
        return output

class BertSelfAttention(nn.Module):

    def __init__(self, d_model: int, num_heads: int):
        super(BertSelfAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads

        self.query = Dense(d_model, d_model)
        self.key = Dense(d_model, d_model)
        self.value = Dense(d_model, d_model)
        self.dropout = Dropout(p=0.1)

    def forward(self, query, key, value, mask):
        n_batches = query.shape[0]

        q = self.query(query)  # (batch_size, seq_len, d_model)
        k = self.key(key)  # (batch_size, seq_len, d_model)
        v = self.value(value)  # (batch_size, seq_len, d_model)

        q = self._split(q, n_batches)  # (batch_size, num_heads, seq_len_q, depth)
        k = self._split(k, n_batches)
        v = self._split(v, n_batches)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = self.sdpa(q, k, v, mask)

        scaled_attention = scaled_attention.transpose(1, 2).contiguous()  # (batch_size, seq_len_q, num_heads, depth)
        # 'Concatenation'
        scaled_attention = scaled_attention.view(n_batches, -1, self.d_model)  # (batch_size, seq_len, d_model)

        return scaled_attention

    def sdpa(self, queries, keys, values, mask):
        """
                Keys and values should have the same length: seq_len_k == seq_len_v.

                :param queries: shape = (..., seq_len_q, d_k)
                :param keys:    shape = (..., seq_len_k, d_k)
                :param values:  shape = (..., seq_len_v, d_v)
                :return:    shape = (..., seq_len_q, seq_len_k)
                """
        matmul_qk = torch.matmul(queries, keys.transpose(-2, -1))  # (..., seq_len_q, seq_len_k)
        d_k = keys.size(-1)
        scaled_attention = torch.divide(matmul_qk, math.sqrt(d_k))

        if mask is not None:
            # Adding a large negative number results
            # in a 0 in the softmax function, thus
            # ignoring padding in the batch.
            scaled_attention += (mask * -1e9)

        attention_weights = softmax(scaled_attention)

        output = torch.matmul(attention_weights, values)

        return output, attention_weights

    def _split(self, x, n_batches):
        return x.view(n_batches, -1, self.num_heads, self.depth).transpose(1, 2)


class BertOutput(nn.Module):

    def __init__(self, in_features: int, out_features: int):
        super(BertOutput, self).__init__()

        self.dense = Dense(in_features, out_features)
        self.LayerNorm = LayerNormalization(out_features)
        self.dropout = Dropout(p=0.1)

    def forward(self, x, residual):
        return self.LayerNorm(residual + self.dense(x))


class BertIntermediate(nn.Module):

    def __init__(self, d_model: int, d_ff: int):
        super(BertIntermediate, self).__init__()

        self.dense = Dense(d_model, d_ff)
        self.activation = Relu()

    def forward(self, x):
        out1 = self.dense(x)  # (batch_size, seq_len, d_ff)
        activation_out = self.activation(out1)
        return activation_out


class Dense(nn.Module):

    def __init__(self, input_dim, hidden_dim: int):
        super(Dense, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.weight = nn.Parameter(torch.zeros([hidden_dim, input_dim]))
        self.bias = nn.Parameter(torch.zeros([hidden_dim]))

        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        a = torch.matmul(x, torch.permute(self.weight, [1, 0])) + self.bias
        return a


class LayerNormalization(nn.Module):
    """
    Layer Normalization: https://arxiv.org/pdf/1607.06450.pdf
    """

    def __init__(self, features, axis=-1, epsilon=1e-05):
        super(LayerNormalization, self).__init__()
        self.axis = axis
        self.epsilon = epsilon

        self.weight = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))

    def forward(self, x):
        dim_size = x.size()[self.axis]

        sum_dim = torch.sum(x, dim=self.axis, keepdim=True)  # (A, B, 1)
        mean_dim = torch.divide(sum_dim, dim_size)  # (A, B, 1)

        pow_dim = torch.pow(x - mean_dim, 2)
        pow_sum = torch.sum(pow_dim, dim=self.axis, keepdim=True)
        var_dim = torch.divide(pow_sum, dim_size)

        x_normalized = torch.divide(x - mean_dim, torch.sqrt(var_dim + self.epsilon))

        return self.weight * x_normalized + self.bias


class Relu(nn.Module):

    def __init__(self):
        super(Relu, self).__init__()

    def forward(self, x):
        return torch.maximum(torch.tensor(0), x)
