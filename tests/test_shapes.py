"""Structural checks for the tiny decoder-only Transformer."""

import unittest

import torch

from src.tiny_transformer import TinyDecoderTransformer


class TestShapes(unittest.TestCase):
    def setUp(self):
        self.model = TinyDecoderTransformer().eval()

    def test_forward_shape(self):
        x = torch.tensor([[0, 1, 2]], dtype=torch.long)
        self.assertEqual(tuple(self.model(x).shape), (1, 3, 5))

    def test_parameter_count(self):
        self.assertEqual(sum(p.numel() for p in self.model.parameters()), 573)

    def test_generate_shape(self):
        x = torch.tensor([[0]], dtype=torch.long)
        self.assertEqual(tuple(self.model.generate(x, new_tokens=3).shape), (1, 4))

    def test_context_length_guard(self):
        with self.assertRaises(ValueError):
            self.model(torch.zeros(1, 17, dtype=torch.long))


if __name__ == "__main__":
    unittest.main()
