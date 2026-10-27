# Token embeddings

Token IDs are integers, not semantic vectors. The learned matrix $E_{\mathrm{token}}\in\mathbb R^{V\times d_{\mathrm{model}}}$ maps each ID to a dense representation. In this experiment $V=5$ and $d_{\mathrm{model}}=4$, so the embedding table contains **20 trainable scalars**.

```python
self.token_embedding = nn.Embedding(vocab_size, d_model)
```

For indices `(1, 3)`, the lookup gives `(1, 3, 4)`. Conceptually, it selects rows from a learnable matrix; it does not require materializing one-hot vectors. Embeddings are optimized as part of the task, rather than imported from pretrained word vectors.

Next: [Positional embeddings](02-positional-embedding.md).
