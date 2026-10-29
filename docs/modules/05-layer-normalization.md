# Layer normalization: controlling the scale of token representations

Our decoder repeatedly transforms token representations through attention and feed-forward networks, adding their outputs through residual connections. Layer normalization prepares those representations for the next computation by centering and scaling their features, followed by a learned affine transformation.

In this chapter we will calculate LayerNorm for a four-dimensional vector, identify its learnable parameters, and follow its use throughout our model.

Prerequisites: means, variances, elementwise operations, and the shape convention `(batch, token, feature)`.

## 1. What is being normalized?

Suppose a hidden tensor has shape

$$
X\in\mathbb{R}^{B\times T\times d}.
$$

Our code uses `nn.LayerNorm(d_model)`. For this configuration, normalization operates across the final feature dimension, independently for every batch item and token position [1].

With $B=1$, $T=3$, and $d=4$, there are three groups of four numbers. Each group gets its own mean and variance:

| Token position | Features used together | Statistics |
| ---: | --- | --- |
| 0 | $X_{0,0,0},\ldots,X_{0,0,3}$ | $\mu_{0,0},\sigma^2_{0,0}$ |
| 1 | $X_{0,1,0},\ldots,X_{0,1,3}$ | $\mu_{0,1},\sigma^2_{0,1}$ |
| 2 | $X_{0,2,0},\ldots,X_{0,2,3}$ | $\mu_{0,2},\sigma^2_{0,2}$ |

The operation preserves shape `(1, 3, 4)`. It does not average over the three token positions or over other examples in the batch.

This distinction also preserves the decoder's information boundary. Normalizing an earlier token never requires reading a future token's representation. A normalization configured to combine the sequence and feature axes would have different behavior and could introduce future information during training.

## 2. The calculation for one token

Write one token representation as $\mathbf{x}=[x_0,\ldots,x_{d-1}]$. Its mean and variance are

$$
\mu=\frac{1}{d}\sum_{j=0}^{d-1}x_j,
\qquad
\sigma^2=\frac{1}{d}\sum_{j=0}^{d-1}(x_j-\mu)^2.
$$

The normalized coordinates are

$$
\hat{x}_j=\frac{x_j-\mu}{\sqrt{\sigma^2+\epsilon}}.
$$

Finally, LayerNorm produces

$$
y_j=\gamma_j\hat{x}_j+\beta_j.
$$

Here $\gamma,\beta\in\mathbb{R}^{d}$ are learned scale and offset vectors. The positive constant $\epsilon$ stabilizes division when the variance is small. In the PyTorch configuration used by our script, its default value is $10^{-5}$ [1].

The denominator in the variance is $d$, rather than $d-1$. This is the variance of the coordinates being normalized, using the convention implemented by LayerNorm. When reproducing it manually, we must select that same convention.

The quantities have different roles:

| Quantity | How it is obtained |
| --- | --- |
| $\mu$ and $\sigma^2$ | Recomputed from the current token's features |
| $\gamma$ and $\beta$ | Learned model parameters |
| $\epsilon$ | Fixed numerical constant |

## 3. Working through four features

Choose the illustrative vector

$$
\mathbf{x}=[1,2,3,4].
$$

The mean is $\mu=2.5$, so the centered vector is

$$
\mathbf{x}-\mu=[-1.5,-0.5,0.5,1.5].
$$

The variance is

$$
\sigma^2=\frac{2.25+0.25+0.25+2.25}{4}=1.25.
$$

With $\epsilon=10^{-5}$, division by $\sqrt{1.25001}$ gives

$$
\hat{\mathbf{x}}\approx[-1.3416,-0.4472,0.4472,1.3416].
$$

Initially, our LayerNorm parameters have $\gamma_j=1$ and $\beta_j=0$ [1], so its output equals this normalized vector.

The centered values have mean zero. Before the learned affine transformation, their normalized variance is

$$
\frac{1}{d}\sum_j\hat{x}_j^2
=\frac{\sigma^2}{\sigma^2+\epsilon}.
$$

It is therefore close to one when variance is large relative to $\epsilon$, rather than exactly one in general. If all input coordinates are equal, the centered vector is zero and the normalized vector is also zero. The final output in that case is $\beta$.

## 4. Why learn another scale and offset?

The learned affine transformation lets subsequent layers receive feature scales and offsets suited to the prediction task. It operates separately on each coordinate.

For example, suppose training produced

$$
\gamma=[1,2,1,0.5],\qquad\beta=[0,0,1,-1].
$$

For our illustrative input, the result would be

$$
\mathbf{y}\approx[-1.3416,-0.8944,1.4472,-0.3292].
$$

The final output need not have zero mean or variance near one. Statements about normalization to zero mean and unit variance refer to the standardized intermediate vector, with the qualification about $\epsilon$ above.

These parameters are shared across batch items and positions within a particular LayerNorm instance. Position 0 and position 1 have different input-dependent statistics, but use the same $\gamma$ and $\beta$ for that layer. Separate LayerNorm instances have their own parameters.

## 5. Why this can help optimization

As representations pass through learned transformations and residual additions, their offsets and scales can change. Centering and scaling give a sublayer a more controlled input representation. Layer normalization was introduced as a method for improving neural-network training [2]. Its benefit in any particular architecture still depends on the rest of the model and training setup.

We can see part of the mechanism directly from the formula. If the same scalar $c$ is added to every coordinate of a token, its mean also increases by $c$, so the centered representation is unchanged.

For positive uniform rescaling by $a$, the standardized coordinates become

$$
\frac{a(x_j-\mu)}{\sqrt{a^2\sigma^2+\epsilon}}.
$$

With zero $\epsilon$ and positive variance, the factor $a$ would cancel exactly. With nonzero $\epsilon$, this scale invariance is approximate when $a^2\sigma^2$ is large relative to $\epsilon$.

Consequently, normalization removes the common feature offset and reduces sensitivity to uniform positive rescaling. It also discards information about those aspects of the input. The learned affine parameters cannot reconstruct an arbitrary input-specific mean or scale that has been removed. In our pre-norm block, the residual path separately carries the original representation forward.

LayerNorm does not guarantee stable gradients or successful training. Learning rate, initialization, architecture, and data still matter. It also does not make each feature statistically independent or force the representation to follow a Gaussian distribution.

## 6. Where normalization appears in our code

Each `TransformerDecoderBlock` creates two normalization layers:

```python
self.norm1 = nn.LayerNorm(d_model)
self.norm2 = nn.LayerNorm(d_model)
```

The forward method is

```python
x = x + self.attn(self.norm1(x))
x = x + self.ffn(self.norm2(x))
return x
```

Writing the intermediate representation explicitly, this is

$$
U=X+\mathrm{Attention}(\mathrm{LN}_1(X)),
$$

$$
Y=U+\mathrm{FFN}(\mathrm{LN}_2(U)).
$$

This ordering is called pre-norm: normalization occurs before the attention or feed-forward sublayer. The representation added through the residual path has not been replaced by its normalized version. Also, the second normalization receives the updated representation $U$.

For comparison, a post-norm attention sublayer would instead have the form

$$
U=\mathrm{LN}(X+\mathrm{Attention}(X)).
$$

These are different computations; moving the same normalization across the residual addition does not preserve the function.

After both decoder blocks, `TinyDecoderTransformer` applies one more LayerNorm before projecting to vocabulary logits:

```python
self.final_norm = nn.LayerNorm(d_model)
self.lm_head = nn.Linear(d_model, vocab_size)
```

```python
return self.lm_head(self.final_norm(x))
```

The final normalization acts on the four hidden features at each position. It does not normalize the five vocabulary logits into probabilities; the output projection has not yet happened at that point.

## 7. Parameters and training behavior

Each of our `LayerNorm(4)` instances has four scale and four offset parameters, giving eight trainable scalars. Across the complete model:

| Location | Instances | Parameters |
| --- | ---: | ---: |
| First decoder block | 2 | 16 |
| Second decoder block | 2 | 16 |
| Final normalization | 1 | 8 |
| Total | 5 | 40 |

These 40 parameters are included in the model's total of 573.

Backpropagation differentiates through the normalization operation, including the dependence of the mean and variance on the input. They are not treated as fixed constants when calculating input gradients.

For example, if $g_{b,t,j}=\partial\mathcal{L}/\partial Y_{b,t,j}$ is the gradient arriving at one LayerNorm output, then

$$
\frac{\partial\mathcal{L}}{\partial\gamma_j}
=\sum_{b,t}g_{b,t,j}\hat{X}_{b,t,j},
\qquad
\frac{\partial\mathcal{L}}{\partial\beta_j}
=\sum_{b,t}g_{b,t,j}.
$$

Input coordinates within one token are coupled through their common statistics. This coupling is confined to the feature axis in our configuration, so it does not create communication between token positions.

## 8. Reproducing LayerNorm manually

The following standalone example compares PyTorch's layer with the same calculation written explicitly:

```python
import torch
from torch import nn

x = torch.tensor([[[1., 2., 3., 4.]]], dtype=torch.float64)
layer = nn.LayerNorm(4, eps=1e-5).double()

mean = x.mean(dim=-1, keepdim=True)
variance = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
standardized = (x - mean) / torch.sqrt(variance + layer.eps)
manual = standardized * layer.weight + layer.bias

print(layer(x))
print(torch.allclose(layer(x), manual))  # True

with torch.no_grad():
    layer.weight.copy_(torch.tensor([1., 2., 1., 0.5]))
    layer.bias.copy_(torch.tensor([0., 0., 1., -1.]))

print(layer(x))
```

`keepdim=True` preserves a final axis of size one, allowing each mean and variance to broadcast back over the four features of its corresponding token. The parameter vectors broadcast over batch and token axes.

The code calculates the variance directly as a mean of squared deviations, avoiding ambiguity about the variance estimator's denominator.

## 9. Training mode, evaluation mode, and other operations

LayerNorm computes its statistics from the current input in both training and evaluation modes. It has no running mean or running variance to substitute during inference [1]. For unchanged parameters and the same input, switching this layer to evaluation mode does not change its calculation.

Other components can still change the input reaching it. In particular, dropout elsewhere in the model behaves differently between training and evaluation. Also, `eval()` does not by itself disable gradient tracking; our generation method uses `torch.no_grad()` for that purpose.

The surrounding operations have distinct responsibilities:

| Operation | Role in our decoder |
| --- | --- |
| LayerNorm | Centers and scales each token's features, then applies learned scale and offset |
| Dropout | Randomly masks and rescales selected activations during training when its probability is nonzero |
| GELU | Applies a pointwise nonlinear activation within the FFN |
| Residual addition | Adds a sublayer's update to the representation carried through the block |

LayerNorm is itself a nonlinear function of its input because its statistics depend on that input. Its purpose and computation nevertheless differ from a pointwise activation such as GELU.

We have now located normalization within the complete decoder computation. The next chapter follows the residual connections and explains how the unchanged path and the learned update interact in both forward computation and backpropagation.

## References and connections

- [1] [PyTorch: `nn.LayerNorm`](https://docs.pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html), for configuration, parameter initialization, and train/eval behavior.
- [2] Ba, Kiros, and Hinton, [Layer Normalization](https://arxiv.org/abs/1607.06450), 2016.
- [Previous chapter: multi-head attention](https://github.com/emiliano-riva/transformer-decoder-only/blob/main/docs/modules/04-multi-head-attention.md).
- [Implementation: decoder blocks and final normalization](https://github.com/emiliano-riva/transformer-decoder-only/blob/main/src/tiny_transformer.py).
- [Next chapter: residual connections](https://github.com/emiliano-riva/transformer-decoder-only/blob/main/docs/modules/06-residual-connections.md).
