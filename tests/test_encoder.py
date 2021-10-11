from unittest import TestCase

import numpy as np
import torch

from src.transformer.encoder import Dense, Embedding, LayerNormalization, MultiHeadAttention, Relu, TransformerEncoder
from src.transformer.operations import create_padding_mask


class TestRelu(TestCase):

    def test_forward_values_correct(self):
        relu = Relu()
        shape = [16, 40, 512]
        tensor = torch.full(shape, fill_value=-1, dtype=torch.float32)
        expected_out = torch.full(shape, fill_value=0, dtype=torch.float32)

        out = relu(tensor)

        torch.testing.assert_allclose(out, expected_out)


class TestDense(TestCase):

    def test_forward(self):
        d_model = 512
        dense = Dense(d_model, d_model)
        shape = [16, 40, d_model]
        tensor = torch.ones(shape, dtype=torch.float32)

        out = dense(tensor)

        torch.testing.assert_equal(out.shape, shape)


class TestLayerNormalization(TestCase):

    def test_forward(self):
        n_features = 2
        layer = LayerNormalization(n_features)
        data = torch.from_numpy(np.arange(10, dtype=np.float32).reshape(-1, n_features) * 10)
        expected_output_np = np.array([-1, 1], dtype=np.float32)
        expected_output_np = np.broadcast_to(expected_output_np, data.shape)
        expected_output = torch.from_numpy(expected_output_np)

        output = layer(data)

        torch.testing.assert_allclose(output, expected_output)


class TestMultiHeadAttention(TestCase):

    def test_forward(self):
        mha = MultiHeadAttention(d_model=512, num_heads=8)
        y_np = np.random.rand(*[1, 60, 512]).astype(np.float32)
        y = torch.from_numpy(y_np)  # (batch_size, encoder_sequence, d_model)
        out, attn = mha(y, y, y, mask=None)

        expected_out_shape = [1, 60, 512]
        # The attentions between each word pair for each head
        expected_attn_shape = [1, 8, 60, 60]

        torch.testing.assert_allclose(out.shape, expected_out_shape)
        torch.testing.assert_allclose(attn.shape, expected_attn_shape)


class TestEmbedding(TestCase):

    def test_forward(self):
        vocab_size = 1000
        d_model = 10
        embedding_layer = Embedding(vocab_size, d_model)
        x = torch.from_numpy(np.random.uniform(0, vocab_size, size=(32, 80)).astype(np.int64))

        embeddings = embedding_layer(x)

        torch.testing.assert_equal(embeddings.shape, [32, 80, d_model])


class TestTransformerEncoder(TestCase):

    def test_forward(self):
        batch_size = 32
        seq_len = 80
        d_model = 512
        encoder = TransformerEncoder(num_layers=1, d_model=d_model, num_heads=4, d_ff=128,
                                     input_vocab_size=5000, max_position_encoding=10000)
        input = torch.from_numpy(np.random.uniform(0, 5000, size=(batch_size, seq_len)).astype(np.int64))
        mask = create_padding_mask(input)
        output = encoder(input, mask)

        torch.testing.assert_equal(output.shape, [batch_size, seq_len, d_model])
