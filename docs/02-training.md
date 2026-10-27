# Training: teacher forcing and next-token prediction

The input–target pairing is created by shifting the training sequence one position to the left:

| Input context ending at | Next-token target |
| --- | --- |
| `il` | `gatto` |
| `il gatto` | `mangia` |
| `il gatto mangia` | `pesce` |

The tensors are `inputs = [[0, 1, 2]]` and `targets = [[1, 2, 3]]`. The Transformer makes three predictions in **one forward pass**; the causal mask prevents a position from reading subsequent inputs. This is *teacher forcing*: during training the observed tokens, rather than model-generated tokens, form the context.

## Cross-entropy

For $T$ target positions, vocabulary size $V$, target indicators $y_{tj}$, and predicted probabilities $p_{tj}$, the training loss is

$$
\mathcal L=-\frac{1}{T}\sum_{t=1}^{T}\sum_{j=1}^{V}y_{tj}\log p_{tj}.
$$

The PyTorch loss `F.cross_entropy` accepts **raw logits** and computes a numerically stable equivalent of log-softmax plus negative log-likelihood. Do not apply softmax explicitly before passing logits to `F.cross_entropy`.

The training loop performs a forward pass, computes loss, clears stale gradients, computes gradients with `loss.backward()`, and updates parameters using AdamW (`lr=0.02`, `weight_decay=0.0`). In this demonstration, the learning rate is deliberately large and should not be used as a general recommendation.

A falling training loss on one three-position example demonstrates memorization of that example only. See [From logits to softmax](deep-dives/from-logits-to-softmax.md) for the probability model and [Backpropagation](deep-dives/backpropagation.md) for the planned gradient derivation.
