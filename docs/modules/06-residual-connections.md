# Residual connections

A residual connection adds a sublayer's output to its input:

$$
y=x+f(x).
$$

With pre-normalization in our decoder block:

$$
u=x+\operatorname{Attention}(\operatorname{LN}_1(x)),
$$

$$
y=u+\operatorname{FFN}(\operatorname{LN}_2(u)).
$$

The addition requires matching tensor shapes. Residual paths provide a direct component in the gradient flow through deeper networks, helping optimization. They do **not** guarantee the absence of vanishing gradients under all circumstances.

Next: [Feed-forward network](07-feed-forward-network.md).
