# Causal self-attention

Self-attention allows a token representation to aggregate information from positions in the same sequence. Autoregressive prediction requires that each position only use its own token and the preceding tokens.

With queries $Q$, keys $K$, and values $V_{\mathrm{att}}$ (not the vocabulary size), scaled dot-product attention is

$$
A=\operatorname{softmax}_{\mathrm{rows}}\left(\frac{QK^\top}{\sqrt{d_{\mathrm{head}}}}+M\right),\qquad C=AV_{\mathrm{att}}.
$$

The causal mask has $M_{ij}=0$ when $j\le i$, and $M_{ij}=-\infty$ when $j>i$. For a three-token sequence:

$$
M=\begin{bmatrix}0 & -\infty & -\infty\\0 & 0 & -\infty\\0 & 0 & 0\end{bmatrix}.
$$

Each row is normalized across **allowed key positions**. Masked future positions get weight zero, so future inputs cannot affect earlier outputs. This property is checked in `tests/test_causal_mask.py`.

Although both this operation and the output layer use softmax, the interpretations differ: attention weights are normalized aggregation weights over positions; output probabilities are a categorical distribution over next tokens.

Next: [Multi-head attention](04-multi-head-attention.md).
