# Learned positional embeddings

Pure self-attention does not inherently distinguish token order without positional information. This implementation learns a position vector $p_t\in\mathbb R^4$ for each position $t\in\{0,\dots,15\}$ and adds it to the token vector:

$$
x_t=e_{w_t}+p_t.
$$

The position table has shape `(16, 4)` and contains **64 trainable scalars**. The input sequence representations retain shape `(B, T, 4)` because addition is elementwise.

```python
positions = torch.arange(length, device=token_ids.device)
x = self.token_embedding(token_ids) + self.position_embedding(positions)[None, :, :]
```

These learned absolute embeddings differ from the sinusoidal encodings proposed in the original Transformer paper and from rotary position embeddings (RoPE) used in many more recent models.

Next: [Causal self-attention](03-causal-self-attention.md).
