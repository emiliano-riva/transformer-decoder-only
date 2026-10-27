# Multi-head attention

Rather than computing one set of attention weights, multi-head attention learns several separate Q/K/V projections and aggregates their outputs. In this experiment $d_{\mathrm{model}}=4$ and the number of heads is $H=2$, so each head operates with $d_{\mathrm{head}}=2$.

For input $X\in\mathbb R^{B\times T\times4}$:

$$
Q=XW_Q^\top,\qquad K=XW_K^\top,\qquad V_{\mathrm{att}}=XW_V^\top.
$$

Each of these tensors is reshaped from `(B, T, 4)` to `(B, 2, T, 2)`; attention produces `(B, 2, T, 2)`. Concatenation restores `(B, T, 4)`, and a learned output projection mixes information across head outputs.

```python
context = weights @ v
context = context.transpose(1, 2).contiguous().reshape(batch, length, d_model)
return self.resid_dropout(self.W_O(context))
```

The same causal restriction applies to every head. Distinct heads **may** learn different patterns; they are not guaranteed to develop interpretable specializations.

Next: [Layer normalization](05-layer-normalization.md).
