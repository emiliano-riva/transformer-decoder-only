# Backpropagation through the decoder-only Transformer

> **Planned deep dive.** This document currently defines the learning objectives and serves as the placeholder for a full mathematical derivation.

The derivation will begin from the per-token categorical cross-entropy, establish $\partial\mathcal L/\partial z=p-y$, and apply the chain rule backward through the vocabulary projection, final normalization, residual paths, feed-forward layers, and attention projections $W_Q$, $W_K$, and $W_V$.

A subsequent worked example will track the shape of each gradient in the tiny $(V=5, d_{\mathrm{model}}=4)$ model and show how `loss.backward()` and `optimizer.step()` act on these parameters.

See [Training](../02-training.md) for the existing executable implementation.
