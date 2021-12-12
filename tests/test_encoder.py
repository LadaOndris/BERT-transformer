from unittest import TestCase

import numpy as np
import torch
from torch.nn import LayerNorm

from src.transformer.modules import Dense, Embedding, LayerNormalization, Relu, TransformerEncoder
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

    def test_layer_against_torch_library(self):
        n_features = 5
        layer = LayerNormalization(n_features)
        torch_layer = LayerNorm(n_features)
        data = torch.from_numpy(np.arange(10, dtype=np.float32).reshape(-1, n_features) * 10)
        # data = torch.tensor([[0, 0, 1, 0, 0.]])
        expected_output = torch_layer(data)
        output = layer(data)

        torch.testing.assert_allclose(output, expected_output)

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
        # Transformer encoder expects words in embeddings
        encoder = TransformerEncoder(num_layers=1, d_model=d_model, num_heads=4, d_ff=128, layer_norm_eps=1e-12)
        input = torch.from_numpy(np.random.uniform(0, 5000, size=(batch_size, seq_len, d_model)).astype(np.float32))
        # Mask doesn't matter
        mask = create_padding_mask(input[:, :, 0])

        output = encoder(input, mask)

        torch.testing.assert_equal(output.shape, [batch_size, seq_len, d_model])
