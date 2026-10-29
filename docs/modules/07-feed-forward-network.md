# Feed-forward network: transforming each token's features

Causal attention lets a token gather information from the positions it is allowed to read. The feed-forward network (FFN) then transforms the features of that contextual representation. In our decoder, it expands four features to sixteen, applies GELU, and projects the result back to four features for a residual addition.

This chapter follows that calculation, explains why the activation matters, and separates its role from dropout and normalization.

Prerequisites: matrix multiplication, affine transformations, and the preceding chapters on attention, layer normalization, and residual connections.

## 1. Where the FFN fits

Recall the two stages of our pre-norm decoder block:

$$
U=X+\mathrm{Attention}(\mathrm{LN}_1(X)),
$$

$$
Y=U+\mathrm{FFN}(\mathrm{LN}_2(U)).
$$

The FFN receives the normalized representation after the attention update. Its output is an update that is added to $U$. Normalization and residual addition surround the FFN; they are not included in `self.ffn` itself.

The implementation is:

```python
self.ffn = nn.Sequential(
    nn.Linear(d_model, d_ff),
    nn.GELU(),
    nn.Dropout(dropout),
    nn.Linear(d_ff, d_model),
    nn.Dropout(dropout),
)
```

Our settings are `d_model=4`, `d_ff=16`, and `dropout=0.0`. The forward method calls this sequence with:

```python
x = x + self.ffn(self.norm2(x))
```

Here `x` already includes the attention update.

## 2. What “position-wise” means

For an input tensor $Z$ of shape `(B, T, 4)`, the FFN applies the same transformation separately to each vector $Z_{b,t,:}$. It does not multiply across the token axis.

| Component | What it combines |
| --- | --- |
| Causal self-attention | Information from permitted token positions, using learned feature projections and input-dependent attention weights |
| FFN | Features within the representation at one position |

The FFN's weights are shared across positions and batch items. They are not shared between our two decoder blocks: each block creates its own FFN instance.

Position-wise processing does not mean that the FFN ignores context. Its input already contains information gathered by attention. It transforms that context without directly reading another position's representation.

This also explains why the FFN needs no causal mask. If its input at position $t$ contains only information from positions up to $t$, a transformation of that vector alone cannot introduce future-token information.

## 3. The two affine transformations

Use row vectors to match PyTorch's calculation. Let $\mathbf{z}\in\mathbb{R}^{1\times4}$ be one normalized token representation. With dropout inactive:

$$
\mathbf{a}=\mathbf{z}W_1^{\mathsf{T}}+\mathbf{b}_1,
$$

$$
\mathbf{h}=\mathrm{GELU}(\mathbf{a}),
$$

$$
\mathbf{r}=\mathbf{h}W_2^{\mathsf{T}}+\mathbf{b}_2.
$$

PyTorch stores a linear layer's weights with shape `(out_features, in_features)` [1]. Our parameter shapes are therefore:

| Parameter | Shape | Trainable scalars |
| --- | --- | ---: |
| $W_1$ | `(16, 4)` | 64 |
| $\mathbf{b}_1$ | `(16,)` | 16 |
| $W_2$ | `(4, 16)` | 64 |
| $\mathbf{b}_2$ | `(4,)` | 4 |
| Total per FFN | | 148 |

Each row of $W_1$ forms one learned combination of the four input features. GELU acts on each of the sixteen resulting coordinates. Each row of $W_2$ then combines those activated features into one output coordinate.

For the full tensor, the shapes are:

$$
(B,T,4)\longrightarrow(B,T,16)\longrightarrow(B,T,16)
\longrightarrow(B,T,4).
$$

Batch size and sequence length remain unchanged. The final width of four makes the residual addition possible.

## 4. Why expand the hidden dimension?

Sixteen intermediate features provide sixteen learned combinations of the input to which the activation can respond. The second layer combines those responses into an update with the required output width.

Expansion alone does not create new input information. Its usefulness comes from the family of nonlinear transformations that the two layers and activation can represent. The intermediate features also need not correspond to individually interpretable concepts.

The ratio $d_{\mathrm{ff}}/d_{\mathrm{model}}=4$ is a choice in this implementation, not a mathematical requirement. Increasing the intermediate width increases parameters and computation; it does not automatically improve a model's predictions.

## 5. Why the activation is necessary

Imagine removing GELU and disabling dropout. The remaining computation would be

$$
\mathbf{r}
=(\mathbf{z}W_1^{\mathsf{T}}+\mathbf{b}_1)W_2^{\mathsf{T}}+\mathbf{b}_2.
$$

Rearranging gives

$$
\mathbf{r}
=\mathbf{z}(W_2W_1)^{\mathsf{T}}
+\mathbf{b}_1W_2^{\mathsf{T}}+\mathbf{b}_2.
$$

This is a single affine transformation. Two consecutive affine layers would not, by themselves, provide a nonlinear FFN.

GELU makes the intermediate response depend nonlinearly on the input, so this general collapse into one affine layer is no longer possible. This argument concerns the FFN in isolation: attention and LayerNorm elsewhere in the decoder are also nonlinear computations.

## 6. GELU and ReLU

The Gaussian Error Linear Unit is defined as [2, 3]

$$
\mathrm{GELU}(a)=a\Phi(a),
$$

where $\Phi$ is the standard normal cumulative distribution function. Equivalently,

$$
\mathrm{GELU}(a)
=\frac{a}{2}\left[1+\mathrm{erf}\left(\frac{a}{\sqrt{2}}\right)\right].
$$

ReLU instead uses $\mathrm{ReLU}(a)=\max(0,a)$.

| Input $a$ | GELU, approximately | ReLU |
| ---: | ---: | ---: |
| -2 | -0.0455 | 0 |
| -1 | -0.1587 | 0 |
| 0 | 0 | 0 |
| 1 | 0.8413 | 1 |
| 2 | 1.9545 | 2 |

GELU smoothly scales each value by $\Phi(a)$. Negative inputs can produce negative outputs; they are not all set to zero. The normal CDF is part of a deterministic formula: GELU does not sample a random gate or require normally distributed inputs.

Our `nn.GELU()` uses the default `approximate='none'`. PyTorch also offers a tanh approximation [2]. GELU remains active during evaluation, because it is part of the learned network's transformation, not a training-only regularizer.

## 7. A numerical example inside the 4 → 16 → 4 network

Take the illustrative FFN input

$$
\mathbf{z}=[1,-1,0,0].
$$

This is an assumed branch input, rather than a claim about the preceding LayerNorm's actual output. Choose $W_1$ so that its first four rows form the $4\times4$ identity matrix and its remaining twelve rows are zero. Set $\mathbf{b}_1=0$.

The first affine transformation gives

$$
\mathbf{a}=[1,-1,0,0,\underbrace{0,\ldots,0}_{12}].
$$

Applying GELU gives

$$
\mathbf{h}\approx[0.8413,-0.1587,0,0,\underbrace{0,\ldots,0}_{12}].
$$

Choose the second layer to produce

$$
r_0=h_0+h_1,\qquad r_1=h_0-h_1,\qquad
r_2=0.5h_0,\qquad r_3=0.1.
$$

The last coordinate comes from its bias. The update is then

$$
\mathbf{r}\approx[0.6827,1.0000,0.4207,0.1000].
$$

These sparse, hand-selected parameters make the calculation readable. A trained FFN generally uses all sixteen intermediate coordinates. The result is the FFN update; the decoder still adds it to the unnormalized residual representation $U$.

This standalone snippet reproduces the example:

```python
import torch
from torch import nn

ffn = nn.Sequential(
    nn.Linear(4, 16), nn.GELU(), nn.Dropout(0.0),
    nn.Linear(16, 4), nn.Dropout(0.0),
).double()

with torch.no_grad():
    for parameter in ffn.parameters():
        parameter.zero_()
    ffn[0].weight[:4].copy_(torch.eye(4, dtype=torch.float64))
    ffn[3].weight[0, :2] = 1.0
    ffn[3].weight[1, 0] = 1.0
    ffn[3].weight[1, 1] = -1.0
    ffn[3].weight[2, 0] = 0.5
    ffn[3].bias[3] = 0.1

z = torch.tensor([[[1., -1., 0., 0.]]], dtype=torch.float64)
print(ffn(z))  # Approximately [[[0.6827, 1.0000, 0.4207, 0.1000]]]
print(sum(p.numel() for p in ffn.parameters()))  # 148
```

## 8. Dropout has a different job

Our FFN contains dropout after GELU and after the second affine transformation. During training, with probability $p$ strictly below one, ordinary dropout acts elementwise as [4]

$$
\mathrm{Dropout}(h_j)=\frac{m_jh_j}{1-p},
\qquad m_j\sim\mathrm{Bernoulli}(1-p).
$$

Some activations become zero; surviving activations are scaled by $1/(1-p)$. For a fixed activation, this preserves its expectation over the dropout mask. Dropout does not permanently remove weights or change the tensor shape.

During evaluation it acts as the identity. In our default run, $p=0$, so both dropout layers act as identities during training as well.

The operations have distinct roles:

| Operation | Role |
| --- | --- |
| GELU | Introduces a deterministic pointwise nonlinearity |
| Dropout | Perturbs activations during training when $p>0$ |
| LayerNorm | Normalizes each token's features, then applies learned scale and offset |
| Residual addition | Adds the FFN update to the representation carried through the block |

## 9. How the FFN learns

Ignoring dropout, use column vectors for a compact derivative calculation:

$$
\mathbf{a}=W_1\mathbf{z}+\mathbf{b}_1,
\qquad \mathbf{h}=\mathrm{GELU}(\mathbf{a}),
\qquad \mathbf{r}=W_2\mathbf{h}+\mathbf{b}_2.
$$

If $\mathbf{g}=\partial\mathcal{L}/\partial\mathbf{r}$, then

$$
\frac{\partial\mathcal{L}}{\partial\mathbf{h}}=W_2^{\mathsf{T}}\mathbf{g},
$$

$$
\boldsymbol{\delta}
=(W_2^{\mathsf{T}}\mathbf{g})\odot\mathrm{GELU}'(\mathbf{a}),
$$

$$
\frac{\partial\mathcal{L}}{\partial\mathbf{z}}
=W_1^{\mathsf{T}}\boldsymbol{\delta}.
$$

Here $\odot$ denotes elementwise multiplication. Differentiating $a\Phi(a)$ gives $\mathrm{GELU}'(a)=\Phi(a)+a\phi(a)$, where $\phi$ is the standard normal density.

For one token's contribution, the parameter gradients are

$$
\frac{\partial\mathcal{L}}{\partial W_2}=\mathbf{g}\mathbf{h}^{\mathsf{T}},
\qquad
\frac{\partial\mathcal{L}}{\partial W_1}=\boldsymbol{\delta}\mathbf{z}^{\mathsf{T}}.
$$

Bias gradients are $\mathbf{g}$ and $\boldsymbol{\delta}$ respectively. Contributions accumulate across batch items and positions because the same parameters are reused. The preceding LayerNorm and the residual path also participate in backpropagation, as described in the previous chapters.

Across our two blocks, the FFNs contain $2\times148=296$ trainable parameters. GELU and dropout add no parameters. These counts describe the FFNs alone, excluding normalization, attention, and the output layer.

The FFN returns hidden features, not vocabulary probabilities. After the decoder stack, final normalization and the output projection convert those features into vocabulary logits. That is the subject of the next chapter.

## References and connections

- [1] [PyTorch: `nn.Linear`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html).
- [2] [PyTorch: `nn.GELU`](https://docs.pytorch.org/docs/stable/generated/torch.nn.GELU.html).
- [3] Hendrycks and Gimpel, [Gaussian Error Linear Units (GELUs)](https://arxiv.org/abs/1606.08415), 2016.
- [4] [PyTorch: `nn.Dropout`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Dropout.html).
- [Previous chapter: residual connections](https://github.com/emiliano-riva/transformer-decoder-only/blob/main/docs/modules/06-residual-connections.md).
- [Implementation: `TransformerDecoderBlock`](https://github.com/emiliano-riva/transformer-decoder-only/blob/main/src/tiny_transformer.py).
- [Next chapter: output layer](https://github.com/emiliano-riva/transformer-decoder-only/blob/main/docs/modules/08-output-layer.md).
