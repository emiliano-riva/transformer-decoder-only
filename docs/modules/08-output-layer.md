# Vocabulary projection and output logits

After the final LayerNorm, a learned linear layer projects the hidden state at each position $h_t\in\mathbb R^4$ to $V=5$ **logits**:

$$
z_t=W_{\mathrm{out}}h_t+b_{\mathrm{out}},\qquad W_{\mathrm{out}}\in\mathbb R^{5\times4}.
$$

The output tensor has shape `(B, T, 5)`. The logits are unrestricted real-valued scores, **not** normalized probabilities.

A softmax defines the categorical next-token distribution:

$$
p_{tj}=P(w_{t+1}=j\mid w_{\le t})=\frac{e^{z_{tj}}}{\sum_{k=1}^{5}e^{z_{tk}}}.
$$

The script feeds logits directly into `F.cross_entropy` for training, and uses an argmax over the last position's logits for greedy generation. For the statistical derivation, see [From logits to softmax](../deep-dives/from-logits-to-softmax.md).
