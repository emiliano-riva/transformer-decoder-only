# Layer normalization

LayerNorm normalizes across the representation dimension for each token independently. For $x\in\mathbb R^d$:

$$
\mu=\frac1d\sum_i x_i,\qquad \sigma^2=\frac1d\sum_i(x_i-\mu)^2,
$$

$$
\operatorname{LN}(x)_i=\gamma_i\frac{x_i-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta_i.
$$

The affine parameters $\gamma$ and $\beta$ are learned. Our decoder blocks use **pre-norm** ordering, applying LayerNorm immediately before each attention or FFN sublayer. A final LayerNorm precedes the vocabulary projection.

Normalization stabilizes the scale of intermediate features; it does not replace the causal mask, an activation function, or regularization.

Next: [Residual connections](06-residual-connections.md).
