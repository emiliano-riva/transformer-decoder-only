"""Tiny decoder-only Transformer implemented explicitly with PyTorch.

Educational settings: vocab=5, d_model=4, heads=2, blocks=2, FFN=16.
The script trains on the single example ``il gatto mangia -> gatto mangia pesce``.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


VOCABULARY = ("il", "gatto", "mangia", "pesce", "<eos>")


class CausalMultiHeadAttention(nn.Module):
    """Multi-head scaled dot-product self-attention with a causal mask."""

    def __init__(self, d_model: int = 4, n_heads: int = 2, dropout: float = 0.0):
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, _ = x.shape
        return x.reshape(batch, length, self.n_heads, self.d_head).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, d_model = x.shape
        q = self._split_heads(self.W_Q(x))
        k = self._split_heads(self.W_K(x))
        v = self._split_heads(self.W_V(x))

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_head)
        # True above the diagonal marks future positions to be hidden.
        future_mask = torch.ones(length, length, dtype=torch.bool, device=x.device).triu(1)
        scores = scores.masked_fill(future_mask, float("-inf"))
        weights = self.attn_dropout(F.softmax(scores, dim=-1))
        context = weights @ v                            # (B, H, T, d_head)
        context = context.transpose(1, 2).contiguous().reshape(batch, length, d_model)
        return self.resid_dropout(self.W_O(context))


class TransformerDecoderBlock(nn.Module):
    """Pre-norm self-attention and feed-forward network with residual paths."""

    def __init__(self, d_model: int = 4, n_heads: int = 2,
                 d_ff: int = 16, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = CausalMultiHeadAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class TinyDecoderTransformer(nn.Module):
    """Autoregressive language model producing one logit vector per position."""

    def __init__(self, vocab_size: int = 5, d_model: int = 4,
                 n_heads: int = 2, n_layers: int = 2,
                 d_ff: int = 16, max_len: int = 16,
                 dropout: float = 0.0):
        super().__init__()
        self.max_len = max_len
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([
            TransformerDecoderBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape (batch, sequence_length)")
        _, length = token_ids.shape
        if not 1 <= length <= self.max_len:
            raise ValueError(f"sequence length must be in [1, {self.max_len}]")

        positions = torch.arange(length, device=token_ids.device)
        x = self.token_embedding(token_ids) + self.position_embedding(positions)[None, :, :]
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.final_norm(x))

    @torch.no_grad()
    def generate(self, token_ids: torch.Tensor, new_tokens: int = 3) -> torch.Tensor:
        """Append tokens using greedy decoding; context uses the latest max_len tokens."""
        if token_ids.ndim != 2 or token_ids.shape[1] == 0:
            raise ValueError("a nonempty (batch, length) token tensor is required")
        for _ in range(new_tokens):
            context = token_ids[:, -self.max_len:]
            logits = self(context)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            token_ids = torch.cat((token_ids, next_token), dim=1)
        return token_ids


def main() -> None:
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyDecoderTransformer().to(device)
    inputs = torch.tensor([[0, 1, 2]], dtype=torch.long, device=device)
    targets = torch.tensor([[1, 2, 3]], dtype=torch.long, device=device)
    print("Device:", device)
    print("Input shape:", tuple(inputs.shape))
    print("Logit shape:", tuple(model(inputs).shape))
    print("Trainable parameters:", sum(p.numel() for p in model.parameters()))

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02, weight_decay=0.0)
    model.train()
    for step in range(501):
        logits = model(inputs)
        loss = F.cross_entropy(logits.reshape(-1, len(VOCABULARY)), targets.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 100 == 0:
            print(f"Step {step:3d} | Loss: {loss.item():.5f}")

    model.eval()
    generated = model.generate(torch.tensor([[0]], device=device), new_tokens=3)[0].tolist()
    print("Generated:", " ".join(VOCABULARY[i] for i in generated))


if __name__ == "__main__":
    main()
