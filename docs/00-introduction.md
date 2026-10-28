# Introduction: what are we trying to understand?

A **decoder-only Transformer** models the distribution of the next token given a context of preceding tokens. Unlike an encoder–decoder Transformer, the language model studied here has no separate encoder and no encoder–decoder cross-attention.

For tokens $w_1,\dots,w_T$, the autoregressive factorization is

$$
P(w_1,\dots,w_T)=\prod_{t=1}^{T}P(w_t\mid w_1,\dots,w_{t-1})
$$

The toy model has no beginning-of-sequence token in its training sample: its first input `il` predicts `gatto`, and so on. It therefore learns only the conditional distributions appearing in this tiny example, not the first unconditional term in the factorization.

The learning pathway follows the computation: tokens $\rightarrow$ embeddings $\rightarrow$ masked attention and feed-forward blocks $\rightarrow$ logits $\rightarrow$ loss or token selection.

[Continue to the architecture overview](01-architecture.md).
