"""Changing future tokens must not change logits for earlier positions."""

import unittest

import torch

from src.tiny_transformer import TinyDecoderTransformer


class TestCausalMask(unittest.TestCase):
    def test_future_token_does_not_change_prefix_logits(self):
        torch.manual_seed(42)
        model = TinyDecoderTransformer(dropout=0.0).eval()
        a = torch.tensor([[0, 1, 2]], dtype=torch.long)
        b = torch.tensor([[0, 1, 4]], dtype=torch.long)
        with torch.no_grad():
            logits_a = model(a)
            logits_b = model(b)
        self.assertTrue(torch.allclose(logits_a[:, :2, :], logits_b[:, :2, :], atol=1e-6))


if __name__ == "__main__":
    unittest.main()
