# Causal self-attention: gathering information from the available context

Token and positional embeddings give each input position a representation. Causal self-attention lets each position combine information from its own representation and those of preceding positions. The combination depends on the input, so the model can use different weights for different sequences.

This chapter develops the computation for one attention head, then locates that computation inside our two-head implementation. The next chapter explains how the heads are constructed and combined.

Prerequisites: matrix multiplication, token and positional embeddings, and the basic idea that softmax converts scores into nonnegative weights summing to one.

## 1. What context is available at each position?

Our training sequence is `il gatto mangia`, with next-token targets `gatto mangia pesce`:

| Position | Input token | Available context | Target |
| ---: | --- | --- | --- |
| 0 | `il` | `il` | `gatto` |
| 1 | `gatto` | `il gatto` | `mangia` |
| 2 | `mangia` | `il gatto mangia` | `pesce` |

All three input tokens are present during training. Nevertheless, the prediction at position 0 must not read the input at position 1: doing so would reveal its target, `gatto`.

The attention output at position $t$ may depend on input positions $0,\ldots,t$. Including position $t$ is valid because its target is the next token. This indexing convention explains why our mask permits the diagonal.

Here, “causal” describes a restriction on information flow through sequence positions. It does not mean that the model performs causal inference about events in the world.

## 2. Queries, keys, and values

Let $U\in\mathbb{R}^{T\times d}$ be the input to attention for one sequence. In our pre-norm block, $U$ is obtained by applying LayerNorm to the current hidden representations. In the first block, those representations originate from the sum of token and positional embeddings.

For one head, we define three learned projections:

$$
Q=UW_Q,\qquad K=UW_K,\qquad V=UW_V,
$$

where, for our head dimensions,

$$
W_Q,W_K,W_V\in\mathbb{R}^{d\times d_h},
\qquad Q,K,V\in\mathbb{R}^{T\times d_h}.
$$

Here $V$ denotes the value matrix, not the vocabulary size. Our model has $d=4$ and $d_h=2$.

The projections give each position three roles:

- Its query is compared with the keys to determine aggregation weights.
- Its key participates in those comparisons when other positions consider it.
- Its value supplies the features that will be aggregated.

These are learned numerical transformations. The search analogy of a query “looking for” matching keys is helpful, but no manually specified linguistic categories are assigned to the vectors.

It is self-attention because queries, keys, and values are derived from the same sequence of representations. The projection parameters are shared across positions, while their outputs depend on the representations at those positions.

Our mathematical matrices multiply row representations on the right. PyTorch's `nn.Linear` stores weights in the transposed convention and applies `input @ weight.T`. In the code, `self.W_Q` also projects to all heads at once; the per-head matrices above describe the corresponding portions of that transformation.

## 3. Comparing each query with the keys

Scaled dot-product attention uses the score matrix [1]

$$
S=\frac{QK^\top}{\sqrt{d_h}}\in\mathbb{R}^{T\times T}.
$$

Entry $S_{t,s}$ compares the query at position $t$ with the key at position $s$:

$$
S_{t,s}=\frac{\mathbf{q}_t\cdot\mathbf{k}_s}{\sqrt{d_h}}.
$$

Rows identify positions receiving information; columns identify positions supplying it. A score is a real number, not yet an aggregation weight. Dot products depend on both direction and magnitude, so they are not generally cosine similarities. Also, $S$ need not be symmetric because queries and keys use different projections.

The scaling factor controls the growth of dot-product variance with dimension. Under the simplifying assumption that all query and key coordinates are independent, zero-mean, and unit-variance, the unscaled dot product has variance $d_h$. Dividing by $\sqrt{d_h}$ gives variance one under those assumptions. This motivates reducing the tendency toward very concentrated softmax outputs as dimension grows [1]. Learned representations need not satisfy those assumptions exactly.

## 4. Removing access to future positions

For three positions, define an additive mask

$$
M=\begin{bmatrix}
0&-\infty&-\infty\\
0&0&-\infty\\
0&0&0
\end{bmatrix}.
$$

The masked score matrix is $S+M$. Allowed entries retain their scores, while future entries become $-\infty$.

Applying softmax across each row gives

$$
A_{t,s}=
\begin{cases}
\displaystyle\frac{\exp(S_{t,s})}{\sum_{r=0}^{t}\exp(S_{t,r})},&s\le t,\\
0,&s>t.
\end{cases}
$$

Thus $A$ is lower triangular and each row sums to one. At position 0, the sole permitted weight is $A_{0,0}=1$.

The mask is applied before softmax so that normalization includes only allowed positions. Setting future weights to zero after an unrestricted softmax would leave the remaining weights summing to less than one unless they were renormalized.

Conceptually, $\exp(-\infty)=0$ explains why masked entries receive zero weight. Numerical implementations of softmax use a stable calculation rather than naively exponentiating arbitrarily large scores. Our mask always permits at least the current position, so it never creates an entirely masked row.

## 5. Combining the value vectors

The output of one head is

$$
C=AV\in\mathbb{R}^{T\times d_h},
$$

or, position by position,

$$
\mathbf{c}_t=\sum_{s=0}^{t}A_{t,s}\mathbf{v}_s.
$$

Each output vector aggregates allowed value vectors using weights calculated from queries and keys. Before attention dropout, it is a convex combination: its coefficients are nonnegative and sum to one.

These attention weights are normalized over sequence positions. The later language-model softmax is normalized over vocabulary entries to predict the next token. An attention weight of 0.7 on `gatto` therefore does not mean a 70% probability that the next token is `gatto`.

## 6. A complete numerical example for one head

Choose the following projected vectors for the three positions:

$$
Q=K=\begin{bmatrix}1&0\\0&1\\1&1\end{bmatrix},
\qquad
V=\begin{bmatrix}1&0\\0&2\\3&1\end{bmatrix}.
$$

These are illustrative values chosen to make the arithmetic visible. They are not extracted from the trained model. Equality of $Q$ and $K$ is only a simplifying choice for this example.

With $d_h=2$,

$$
S=\frac{1}{\sqrt{2}}
\begin{bmatrix}1&0&1\\0&1&1\\1&1&2\end{bmatrix}.
$$

Masking gives

$$
S+M\approx
\begin{bmatrix}
0.7071&-\infty&-\infty\\
0&0.7071&-\infty\\
0.7071&0.7071&1.4142
\end{bmatrix}.
$$

Row-wise softmax produces

$$
A\approx
\begin{bmatrix}
1&0&0\\
0.3302&0.6698&0\\
0.2483&0.2483&0.5035
\end{bmatrix}.
$$

The second output is approximately

$$
\mathbf{c}_1=0.3302[1,0]+0.6698[0,2]\approx[0.3302,1.3395].
$$

Using the unrounded weights for the full multiplication gives

$$
C\approx\begin{bmatrix}
1&0\\
0.3302&1.3395\\
1.7587&1.0000
\end{bmatrix}.
$$

Position 0 retains only its own value vector. Position 1 combines the first two values. Position 2 can combine all three. These outputs are intermediate features; the remaining decoder computations and output projection are still needed to produce vocabulary logits.

## 7. The same calculation in Python

```python
import math
import torch

q = torch.tensor([[1., 0.], [0., 1.], [1., 1.]])
k = q.clone()
v = torch.tensor([[1., 0.], [0., 2.], [3., 1.]])

scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
future_mask = torch.ones(3, 3, dtype=torch.bool).triu(1)
masked_scores = scores.masked_fill(future_mask, float("-inf"))
weights = torch.softmax(masked_scores, dim=-1)
context = weights @ v

print(weights)
print(context)
```

`triu(1)` marks entries strictly above the diagonal. `masked_fill` replaces those entries with negative infinity. `dim=-1` applies softmax across keys for each query.

The repository implements this computation with batch and head axes:

| Quantity | Shape | Our laboratory |
| --- | --- | --- |
| `q`, `k`, `v` | `(B, H, T, d_h)` | `(1, 2, 3, 2)` |
| `scores`, `weights` | `(B, H, T, T)` | `(1, 2, 3, 3)` |
| `future_mask` | `(T, T)` | `(3, 3)` |
| `context` before combining heads | `(B, H, T, d_h)` | `(1, 2, 3, 2)` |

Matrix multiplication acts on the last two dimensions, independently for each batch item and head. The same causal mask broadcasts across both leading axes.

Our code applies attention dropout immediately after softmax. With the configured probability `dropout=0.0`, the weights used in `weights @ v` retain the normalization described above. With nonzero dropout during training, some weights are zeroed and the survivors are rescaled; an individual resulting row need not sum to one. The convex-combination description then applies to the weights before dropout.

## 8. Parallel training and sequential generation

During training, all input positions are known. We can compute all queries, keys, and values together and apply the mask to enforce each prediction's information boundary. Parallel computation does not imply access to future tokens.

The same boundary is preserved through our stack. At every block, attention only reads earlier or current positions; LayerNorm and the FFN operate separately on each position; residual additions combine representations at the same position. Consequently, later layers do not introduce a path from a future input to an earlier output.

During generation, the next token is initially unknown. The model processes the available prefix, uses the last position's logits to select a token, appends it, and runs again. The mask and the autoregressive loop therefore address different aspects of the computation.

Our test in `tests/test_causal_mask.py` changes the third input token and checks that the logits at the first two positions stay equal within numerical tolerance. Evaluation mode removes dropout randomness from that comparison. This is a useful check of the implementation for the chosen inputs, alongside the architectural reasoning above.

## 9. What is learned, and what depends on the input?

The projection matrices are trainable parameters. The queries, keys, values, scores, and attention weights are intermediate quantities recomputed for each input. The causal mask is a fixed rule derived from sequence length.

Gradients from the prediction loss flow through the value aggregation and through the softmax weights into the projections and earlier representations. This lets training adjust both the features that positions contribute and the comparisons used to weight those contributions.

Attention weights describe this particular aggregation operation. They should not automatically be treated as a complete explanation of the final prediction, which also depends on value vectors, other heads, residual paths, and subsequent layers.

The next chapter shows how our implementation carries out two such computations in parallel, splits the projected features into heads, concatenates the resulting outputs, and applies the learned output projection.

## References and connections

- [1] Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762), Section 3.2.1, for scaled dot-product attention and the scaling motivation.
- [Previous chapter: positional embeddings](https://github.com/emiliano-riva/transformer-decoder-only/blob/main/docs/modules/02-positional-embedding.md).
- [Implementation: `CausalMultiHeadAttention`](https://github.com/emiliano-riva/transformer-decoder-only/blob/main/src/tiny_transformer.py).
- [Causal-mask test](https://github.com/emiliano-riva/transformer-decoder-only/blob/main/tests/test_causal_mask.py).
- [Next chapter: multi-head attention](https://github.com/emiliano-riva/transformer-decoder-only/blob/main/docs/modules/04-multi-head-attention.md).
