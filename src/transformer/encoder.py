import torch
from torch import nn

from src.transformer.operations import softmax, positional_encoding


class TransformerEncoder(nn.Module):

    def __init__(self, num_layers: int, d_model: int, num_heads: int, d_ff: int, input_vocab_size: int,
                 max_position_encoding: int):
        super(TransformerEncoder, self).__init__()
        self.num_layers = num_layers

        self.embedding = Embedding(input_vocab_size, d_model)
        self.positional_encoding = positional_encoding(max_position_encoding, d_model)

        self.encoder_layers = [EncoderLayer(d_model, num_heads, d_ff)
                               for _ in range(num_layers)]

    def forward(self, x):
        seq_len = x.shape[1]

        x = self.embedding(x)
        x += self.positional_encoding[:, :seq_len, :]

        for i in range(self.num_layers):
            x = self.encoder_layers[i](x)
        return x



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

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = PointWiseFeedForward(d_model, d_ff)
        self.mha_norm = LayerNormalization(d_model)
        self.ffn_norm = LayerNormalization(d_model)

    def forward(self, x):
        attention_out = self.mha(x, x, x)  # (batch_size, seq_len, d_model)
        attention_norm = self.mha_norm(x + attention_out)  # (batch_size, seq_len, d_model)

        ffn_out = self.ffn(attention_norm)  # (batch_size, seq_len, d_model)
        ffn_norm = self.ffn_norm(attention_norm + ffn_out)  # (batch_size, seq_len, d_model)

        return ffn_norm


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model: int, num_heads: int):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads

        self.w_q = Dense(d_model)
        self.w_k = Dense(d_model)
        self.w_v = Dense(d_model)
        self.w_o = Dense(d_model)

        self.sdpa = ScaledDotProductAttention()

    def forward(self, query, key, value):
        n_batches = query.shape[0]

        q = self.w_q(query)  # (batch_size, seq_len, d_model)
        k = self.w_k(key)  # (batch_size, seq_len, d_model)
        v = self.w_v(value)  # (batch_size, seq_len, d_model)

        q = self._split(q, n_batches)  # (batch_size, num_heads, seq_len_q, depth)
        k = self._split(k, n_batches)
        v = self._split(v, n_batches)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = self.sdpa(q, k, v)

        scaled_attention = scaled_attention.transpose(1, 2).contiguous()  # (batch_size, seq_len_q, num_heads, depth)
        # 'Concatenation'
        scaled_attention = scaled_attention.view(n_batches, -1, self.d_model)  # (batch_size, seq_len, d_model)

        output = self.w_o(scaled_attention)  # (batch_size, seq_len, d_model)

        return output, attention_weights

    def _split(self, x, n_batches):
        return x.view(n_batches, -1, self.num_heads, self.depth).transpose(1, 2)


class ScaledDotProductAttention(nn.Module):

    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, queries, keys, values):
        """
        Keys and values should have the same length: seq_len_k == seq_len_v.

        :param queries: shape = (..., seq_len_q, d_k)
        :param keys:    shape = (..., seq_len_k, d_k)
        :param values:  shape = (..., seq_len_v, d_v)
        :return:    shape = (..., seq_len_q, seq_len_k)
        """
        matmul_qk = torch.matmul(queries, keys.transpose(-2, -1))  # (..., seq_len_q, seq_len_k)
        d_k = keys.size()[-1]
        scaled_attention = torch.divide(matmul_qk, d_k)

        attention_weights = softmax(scaled_attention)

        output = torch.matmul(attention_weights, values)

        return output, attention_weights


class PointWiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super(PointWiseFeedForward, self).__init__()

        self.dense1 = Dense(d_ff, activation='relu')
        self.dense2 = Dense(d_model)

    def forward(self, x):
        out1 = self.dense1(x)  # (batch_size, seq_len, d_ff)
        out2 = self.dense2(out1)  # (batch_size, seq_len, d_model)
        return out2


class Dense(nn.Module):

    def __init__(self, hidden_dim: int, activation: str = None):
        super(Dense, self).__init__()
        self.hidden_dim = hidden_dim
        if activation == 'relu':
            self.activation = Relu()
        else:
            self.activation = None

        # TODO: weights and biases intitializer
        self.weights = nn.Parameter(torch.ones([hidden_dim]))
        self.biases = nn.Parameter(torch.ones([hidden_dim]))

    def forward(self, x):
        a = x * self.weights + self.biases
        if self.activation is not None:
            return self.activation(a)
        return a


class LayerNormalization(nn.Module):
    """
    Layer Normalization: https://arxiv.org/pdf/1607.06450.pdf
    """

    def __init__(self, features, axis=-1, epsilon=1e-3):
        super(LayerNormalization, self).__init__()
        self.axis = axis
        self.epsilon = epsilon

        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))

    def forward(self, x):
        dim_size = x.size()[self.axis]

        sum_dim = torch.sum(x, dim=self.axis, keepdim=True)  # (A, B, 1)
        mean_dim = torch.divide(sum_dim, dim_size)  # (A, B, 1)

        pow_dim = torch.pow(x - mean_dim, 2)
        var_dim = torch.divide(pow_dim, dim_size - 1)

        x_normalized = torch.divide(x - mean_dim, torch.sqrt(var_dim + self.epsilon))

        return self.gamma * x_normalized + self.beta


class Relu(nn.Module):

    def __init__(self):
        super(Relu, self).__init__()

    def forward(self, x):
        return torch.maximum(torch.tensor(0), x)
