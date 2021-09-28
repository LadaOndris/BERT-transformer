from unittest import TestCase

import numpy as np
import torch

from src.transformer.encoder import Dense, LayerNormalization, MultiHeadAttention, Relu


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
        dense = Dense(d_model)
        shape = [16, 40, d_model]
        tensor = torch.ones(shape, dtype=torch.float32)

        out = dense(tensor)

        torch.testing.assert_equal(out.shape, shape)


class TestLayerNormalization(TestCase):

    def test_forward(self):
        layer = LayerNormalization()
        data = torch.from_numpy(np.arange(10, dtype=np.float32).reshape(5, 2) * 10)
        expected_output_np = np.array([-1, 1], dtype=np.float32)
        expected_output_np = np.broadcast_to(expected_output_np, data.shape)
        expected_output = torch.from_numpy(expected_output_np)

        output = layer(data)

        torch.testing.assert_allclose(output, expected_output)


class TestMultiHeadAttention(TestCase):

    def test_forward(self):
        mha = MultiHeadAttention(d_model=512, num_heads=8)
        y = torch.from_numpy(np.random.rand(*[1, 60, 512]))  # (batch_size, encoder_sequence, d_model)
        out, attn = mha(y, y, y)

        expected_out_shape = [1, 60, 512]
        # The attentions between each word pair for each head
        expected_attn_shape = [1, 8, 60, 60]

        torch.testing.assert_allclose(out.shape, expected_out_shape)
        torch.testing.assert_allclose(attn.shape, expected_attn_shape)
