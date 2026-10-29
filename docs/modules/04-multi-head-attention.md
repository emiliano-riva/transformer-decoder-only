# Multi-head attention: parallel context aggregation and learned mixing

The previous chapter followed one causal attention head from queries, keys, and values to a weighted combination of the available context. Our implementation performs two such computations, using different learned projections, and combines their outputs into a single representation at each position.

This chapter explains how the mathematical heads correspond to the code: projecting all features at once, introducing a head axis, computing attention, restoring the token axis order, and applying an output projection.

Prerequisites: the causal self-attention chapter, matrix multiplication, and tensor shapes. Throughout, $B$ is batch size, $T$ is sequence length, $d$ is model width, and $H$ is the number of heads.

## 1. Why use more than one head?

For a single head, each query position produces one vector of attention weights over the allowed key positions. All value coordinates in that head are aggregated using those same weights.

Multiple heads allow several independently parameterized aggregation patterns at each position. One head can assign strong weight to the current position while another combines information from earlier positions. Each head also has its own value projection, so the features being aggregated can differ.

These possibilities motivate multi-head attention [1]. They do not guarantee that heads will acquire distinct, easily named linguistic roles. Our tiny training sequence is especially insufficient for establishing such specialization.

In our implementation,

$$
d=4,\qquad H=2,\qquad d_h=\frac{d}{H}=2.
$$

Each head has two-dimensional queries, keys, and values. Both heads process all sequence positions, subject to the same causal restriction.

## 2. Separate projections, shared input

For one sequence, let $U\in\mathbb{R}^{T\times d}$ denote the normalized representations passed into attention. For head $h$, define

$$
Q^{(h)}=UW_Q^{(h)},\qquad
K^{(h)}=UW_K^{(h)},\qquad
V^{(h)}=UW_V^{(h)},
$$

with

$$
W_Q^{(h)},W_K^{(h)},W_V^{(h)}\in\mathbb{R}^{d\times d_h}.
$$

In the two-head model, each of these mathematical projection matrices has shape $4\times2$. Every projected coordinate can depend on all four input coordinates. A head is therefore not limited to a predetermined slice of the input embedding.

For example, one head might compute a query coordinate using

$$
q_{t,0}^{(h)}=u_{t,0}w_{0,0}^{(h)}+
u_{t,1}w_{1,0}^{(h)}+
u_{t,2}w_{2,0}^{(h)}+
u_{t,3}w_{3,0}^{(h)}.
$$

The weights determine that combination during training. Splitting into heads takes place after projection in our code.

## 3. Why the code creates four-by-four matrices

The constructor contains:

```python
self.W_Q = nn.Linear(d_model, d_model, bias=False)
self.W_K = nn.Linear(d_model, d_model, bias=False)
self.W_V = nn.Linear(d_model, d_model, bias=False)
self.W_O = nn.Linear(d_model, d_model, bias=False)
```

We can collect the per-head query matrices into one wider matrix:

$$
W_Q=\begin{bmatrix}W_Q^{(0)}&W_Q^{(1)}\end{bmatrix}
\in\mathbb{R}^{4\times4}.
$$

Then

$$
UW_Q=\begin{bmatrix}UW_Q^{(0)}&UW_Q^{(1)}\end{bmatrix}
=\begin{bmatrix}Q^{(0)}&Q^{(1)}\end{bmatrix}.
$$

Computing all heads with a single projection is thus equivalent to computing their projections separately and joining the resulting feature coordinates. The same reasoning applies to keys and values.

These equations use matrices that multiply row representations on the right. PyTorch stores the transpose: `nn.Linear` computes `input @ weight.T`. Accordingly, the per-head blocks occupy columns of the mathematical $W_Q$ above and corresponding rows of `self.W_Q.weight`.

The parameters assigned to the two heads are distinct parts of the full matrix. Using one `nn.Linear` object does not make those parts identical or tie their weights together.

## 4. Splitting the projected features into heads

After `self.W_Q(x)`, the projected queries have shape `(B, T, d)`. The helper method reorganizes them:

```python
def _split_heads(self, x):
    batch, length, _ = x.shape
    return x.reshape(
        batch, length, self.n_heads, self.d_head
    ).transpose(1, 2)
```

The two operations have different purposes:

| Operation | Shape | Meaning |
| --- | --- | --- |
| Projection output | `(B, T, d)` | All projected features for each token |
| `reshape` | `(B, T, H, d_h)` | Group the projected features by head |
| `transpose(1, 2)` | `(B, H, T, d_h)` | Place sequence and feature axes last for per-head matrix operations |

For our example, the shapes are

$$
(1,3,4)\longrightarrow(1,3,2,2)\longrightarrow(1,2,3,2).
$$

If a token's projected query is `[a, b, c, d]`, head 0 receives `[a, b]` and head 1 receives `[c, d]`. Each of these values was already computed from the full input vector.

Neither reshaping nor transposing introduces trainable parameters. They organize existing values for the next calculation.

The constructor checks that `d_model` is divisible by `n_heads`. This is required by our choice of equal head widths whose concatenation has width `d_model`; it is a constraint of this implementation rather than a universal requirement for every possible attention architecture.

## 5. Causal attention runs independently within each head

For each head, the previous chapter's calculation becomes

$$
A^{(h)}=\mathrm{softmax}_{\mathrm{rows}}\left(
\frac{Q^{(h)}(K^{(h)})^\top}{\sqrt{d_h}}+M
\right),
\qquad C^{(h)}=A^{(h)}V^{(h)}.
$$

This formula describes the attention weights before dropout. In our configured model, dropout is zero.

The implementation computes every batch item and head together:

```python
scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_head)
future_mask = torch.ones(
    length, length, dtype=torch.bool, device=x.device
).triu(1)
scores = scores.masked_fill(future_mask, float("-inf"))
weights = self.attn_dropout(F.softmax(scores, dim=-1))
context = weights @ v
```

Scores and weights have shape `(B, H, T, T)`. Softmax normalizes over allowed key positions separately for every query, head, and batch item. It does not normalize across the head axis.

The causal mask broadcasts over batch and head axes. Each head has its own attention weights, but the same set of allowed positions. The result `context` has shape `(B, H, T, d_h)`.

## 6. Concatenating the outputs in the correct order

We want each token to receive the outputs of both heads:

$$
Z=\begin{bmatrix}C^{(0)}&C^{(1)}\end{bmatrix}
\in\mathbb{R}^{T\times d}.
$$

For illustration, suppose the head outputs for one sequence are

$$
C^{(0)}=\begin{bmatrix}1&2\\3&4\\5&6\end{bmatrix},
\qquad
C^{(1)}=\begin{bmatrix}10&20\\30&40\\50&60\end{bmatrix}.
$$

These chosen values illustrate tensor organization, rather than results from the trained model. Concatenation along features gives

$$
Z=\begin{bmatrix}
1&2&10&20\\
3&4&30&40\\
5&6&50&60
\end{bmatrix}.
$$

Our code obtains that arrangement using

```python
context = context.transpose(1, 2).contiguous().reshape(
    batch, length, d_model
)
```

The transpose changes `(B, H, T, d_h)` into `(B, T, H, d_h)`. Now both head outputs for each token are adjacent in the logical indexing order, and the final reshape combines the last two axes.

Directly reshaping the original `(1, 2, 3, 2)` tensor into `(1, 3, 4)` would instead produce

$$
\begin{bmatrix}
1&2&3&4\\
5&6&10&20\\
30&40&50&60
\end{bmatrix}.
$$

That arrangement mixes different token positions into the same row. The shape is correct, but the meaning of its entries is wrong. Shape checks alone would not catch this mistake.

Transposing generally changes tensor strides without physically rearranging storage. `contiguous()` provides contiguous storage in the transposed order, copying when necessary [2]. `reshape` itself can return a view or make a copy, so the explicit `contiguous()` is not always required when using `reshape` [3]. What is essential here is the correct axis order before combining the head and feature dimensions.

## 7. The output projection mixes head contributions

Concatenation preserves the individual head outputs as feature blocks. The learned output projection then combines them:

$$
Y=ZW_O,\qquad W_O\in\mathbb{R}^{d\times d}.
$$

This operates independently at each token position. It mixes features across heads; the cross-position aggregation has already occurred inside each head.

For a concrete example, choose the mathematical matrix

$$
W_O=\begin{bmatrix}
1&0&1&0\\
0&1&0&1\\
1&0&-1&0\\
0&1&0&-1
\end{bmatrix}.
$$

Then a row $[a,b,c,d]$ becomes $[a+c,b+d,a-c,b-d]$. Applied to our concatenated outputs, this gives

$$
Y=\begin{bmatrix}
11&22&-9&-18\\
33&44&-27&-36\\
55&66&-45&-54
\end{bmatrix}.
$$

In the real model, $W_O$ is learned rather than chosen by hand. It can weight, combine, or suppress head contributions. Concatenation followed by this projection is more flexible than directly averaging the head outputs.

In our code the operation is `self.W_O(context)`, followed by `self.resid_dropout`. The residual addition itself happens outside the attention class, in `TransformerDecoderBlock`.

## 8. Reproducing the merge and projection

This standalone example follows the two numerical matrices above:

```python
import torch

# Axes: batch, head, token, feature.
context = torch.tensor([[
    [[1., 2.], [3., 4.], [5., 6.]],
    [[10., 20.], [30., 40.], [50., 60.]],
]])

merged = context.transpose(1, 2).contiguous().reshape(1, 3, 4)
wrong_merge = context.reshape(1, 3, 4)

# Mathematical right-multiplication convention.
w_o = torch.tensor([
    [1., 0.,  1.,  0.],
    [0., 1.,  0.,  1.],
    [1., 0., -1.,  0.],
    [0., 1.,  0., -1.],
])
output = merged @ w_o

print(merged)
print(wrong_merge)
print(output)
```

To reproduce this same projection with `nn.Linear(4, 4, bias=False)`, its stored weight would be set to `w_o.T`, following PyTorch's convention.

## 9. Parameter count and computational cost

Our attention module contains four bias-free $4\times4$ matrices:

| Projection | Parameters |
| --- | ---: |
| Queries | 16 |
| Keys | 16 |
| Values | 16 |
| Output | 16 |
| Total | 64 |

For fixed model width $d$, the same construction has $4d^2$ projection parameters, whether it uses one head of width $d$ or several heads whose widths sum to $d$. Increasing the number of heads does not multiply that parameter count when model width stays fixed.

It does change how the representation capacity is partitioned. More heads mean narrower per-head queries, keys, and values under this design, and do not automatically imply better results.

The dense attention score tensor contains $BHT^2$ entries. The leading arithmetic cost of the score products is proportional to $BHT^2d_h=BT^2d$; value aggregation has the same order. The projection operations cost on the order of $BTd^2$. These expressions omit constant factors, softmax, and implementation overhead, so equal leading-order costs do not imply identical runtime. In our explicit implementation, every head's full score matrix is materialized, including entries later masked out.

## 10. Returning to the decoder block

The complete attention module preserves shape:

$$
(B,T,d)\longrightarrow(B,T,d).
$$

This makes its output compatible with the residual addition:

```python
x = x + self.attn(self.norm1(x))
```

Within attention, head computations use separate projected coordinates. The output projection combines their contributions, and gradients from the same final prediction loss train all those projections jointly. Each decoder block owns a separate attention module; the two blocks in our model do not share projection weights.

We have now connected single-head causal attention to the complete `CausalMultiHeadAttention` class. The next chapter examines the LayerNorm operation that prepares representations before attention and before the feed-forward network.

## References and connections

- [1] Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762), Section 3.2.2, for multi-head attention.
- [2] [PyTorch: `Tensor.contiguous`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.contiguous.html).
- [3] [PyTorch: `torch.reshape`](https://docs.pytorch.org/docs/stable/generated/torch.reshape.html).
- [Previous chapter: causal self-attention](https://github.com/emiliano-riva/transformer-decoder-only/blob/main/docs/modules/03-causal-self-attention.md).
- [Implementation: `CausalMultiHeadAttention`](https://github.com/emiliano-riva/transformer-decoder-only/blob/main/src/tiny_transformer.py).
- [Next chapter: layer normalization](https://github.com/emiliano-riva/transformer-decoder-only/blob/main/docs/modules/05-layer-normalization.md).
