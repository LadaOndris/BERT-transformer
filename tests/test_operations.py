from unittest import TestCase

import torch

from src.transformer.operations import softmax


class Test(TestCase):
    def test_softmax(self):
        data = torch.ones(size=[16, 8, 4])
        expected_output = torch.full_like(data, fill_value=0.25, dtype=torch.float32)

        output = softmax(data, dim=-1)

        torch.testing.assert_allclose(output, expected_output)
