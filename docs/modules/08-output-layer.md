# Output layer: from hidden representations to next-token predictions

The decoder stack produces a contextual representation at every input position. These vectors have four features in our model, but our vocabulary contains five tokens. The output layer connects these two spaces: it maps each hidden vector to five scores, one for each possible next token.

This chapter follows the output projection, converts its scores into probabilities, and connects those probabilities to training and generation. It completes our tour of the main architectural modules.

Prerequisites: affine transformations, exponentials and logarithms, and the distinction between token positions and vocabulary indices.

## 1. The final normalization and projection

After the two decoder blocks, `TinyDecoderTransformer` applies:

```python
return self.lm_head(self.final_norm(x))
```

The components were created as:

```python
self.final_norm = nn.LayerNorm(d_model)
self.lm_head = nn.Linear(d_model, vocab_size)
```

The name `lm_head` means language-model head. With our settings, it is an affine transformation from four hidden features to five vocabulary scores.

Let $X$ be the final decoder output and $H=\mathrm{LN}_{\mathrm{final}}(X)$. For one position, write the normalized representation as a column vector $\mathbf{h}_t\in\mathbb{R}^4$. The head computes

$$
\mathbf{z}_t=W_{\mathrm{out}}\mathbf{h}_t+\mathbf{b}_{\mathrm{out}},
$$

where

$$
W_{\mathrm{out}}\in\mathbb{R}^{5\times4},
\qquad \mathbf{b}_{\mathrm{out}}\in\mathbb{R}^{5}.
$$

PyTorch applies the same transformation to every position. With row-oriented tensors, the equivalent expression is `H @ W_out.T + b_out`.

| Stage | Shape |
| --- | --- |
| Final decoder output | `(B, T, 4)` |
| Final LayerNorm output | `(B, T, 4)` |
| Vocabulary logits | `(B, T, 5)` |

The head has $5\times4+5=25$ trainable scalars. Final LayerNorm has another eight, counted separately. The head's parameters are shared across positions and batch items.

Our input embedding and output weight matrix both have shape `(5, 4)`, but they are separate parameters in this implementation. Matching shapes do not imply weight sharing or that the output operation inverts the embedding lookup.

## 2. What a logit represents

The vocabulary order is:

| Index | Token |
| ---: | --- |
| 0 | `il` |
| 1 | `gatto` |
| 2 | `mangia` |
| 3 | `pesce` |
| 4 | `<eos>` |

Coordinate $j$ of $\mathbf{z}_t$ is the score assigned to vocabulary token $j$ as the next token after position $t$.

A logit can be positive, negative, or zero. The five scores do not need to sum to one. Each score is a learned weighted combination of the contextual features plus a bias:

$$
z_{t,j}=\sum_{r=0}^{3}W_{\mathrm{out},j,r}h_{t,r}+b_{\mathrm{out},j}.
$$

Although the output head is affine, the complete mapping from the input sequence to logits is nonlinear because the hidden representation was computed by the decoder.

## 3. Softmax defines the next-token distribution

Softmax normalizes over the vocabulary axis [1]:

$$
p_{t,j}=\frac{\exp(z_{t,j})}{\sum_{k=0}^{4}\exp(z_{t,k})}.
$$

If the input tokens are $w_0,\ldots,w_t$, this distribution models

$$
p_{t,j}=P(w_{t+1}=j\mid w_0,\ldots,w_t).
$$

For finite logits, its entries are positive and sum to one in exact arithmetic. Floating-point implementations can underflow extremely small probabilities to zero.

The vocabulary axis is the last axis of `(B, T, 5)`, so probabilities can be inspected with:

```python
probabilities = torch.softmax(logits, dim=-1)
```

This is a different use of softmax from attention. Attention normalizes scores over allowed token positions to obtain aggregation weights. The output softmax normalizes over vocabulary choices to obtain next-token probabilities.

Only differences between logits determine the probabilities:

$$
\log\frac{p_{t,i}}{p_{t,j}}=z_{t,i}-z_{t,j}.
$$

Adding the same constant to every logit leaves the distribution unchanged. Multiplying all logits by a constant generally changes it. For numerical stability, softmax can subtract the maximum score before exponentiating:

$$
p_{t,j}=\frac{\exp(z_{t,j}-m_t)}{\sum_{k=0}^{4}\exp(z_{t,k}-m_t)},
\qquad m_t=\max_k z_{t,k}.
$$

## 4. A complete numerical example

Suppose one post-normalization hidden vector is

$$
\mathbf{h}=[1,0,0,0]^{\mathsf{T}}.
$$

For an illustrative projection, set the first column of $W_{\mathrm{out}}$ to $[0,1,2,3,-1]^{\mathsf{T}}$, all other entries to zero, and the bias to zero. Then

$$
\mathbf{z}=[0,1,2,3,-1]^{\mathsf{T}}.
$$

These are chosen values for explaining the output head, not results from the trained decoder. Softmax gives:

| Token | Logit | Probability, approximately |
| --- | ---: | ---: |
| `il` | 0 | 0.0317 |
| `gatto` | 1 | 0.0861 |
| `mangia` | 2 | 0.2341 |
| `pesce` | 3 | 0.6364 |
| `<eos>` | -1 | 0.0117 |

The most probable next token is `pesce`, but the distribution assigns some probability to every vocabulary entry. The values sum to one before rounding.

If the correct target is `pesce`, its negative log-probability is

$$
\ell=-\log p_3\approx-\log(0.6364)\approx0.4519.
$$

This uses the natural logarithm. A higher probability for the correct target gives a lower loss; selecting the largest score alone does not determine the loss.

## 5. One prediction at every training position

Our training example uses:

```python
inputs = torch.tensor([[0, 1, 2]], dtype=torch.long)
targets = torch.tensor([[1, 2, 3]], dtype=torch.long)
```

The input and target sequences are shifted by one token:

| Position | Input token | Context available through causal attention | Target |
| ---: | --- | --- | --- |
| 0 | `il` | `il` | `gatto` |
| 1 | `gatto` | `il gatto` | `mangia` |
| 2 | `mangia` | `il gatto mangia` | `pesce` |

One forward pass returns `(1, 3, 5)` logits, representing three separate next-token predictions. The model does not use its prediction at position 0 as the input at position 1 during this training pass. It receives the supplied target-sequence prefix, an arrangement called teacher forcing.

The causal mask prevents an earlier position from reading a later input token even though all three positions are processed in the same forward pass.

## 6. Cross-entropy takes logits directly

The implementation calculates:

```python
loss = F.cross_entropy(
    logits.reshape(-1, len(VOCABULARY)),
    targets.reshape(-1),
)
```

Flattening combines the batch and sequence axes. Here logits become `(3, 5)` and targets become `(3,)`. Each row remains paired with its corresponding target index.

For $N=BT$ positions, with no padding or ignored targets, the objective is

$$
\mathcal{L}=-\frac{1}{N}\sum_{n=1}^{N}\log p_{n,y_n}.
$$

The default reduction averages these position losses. PyTorch expects unnormalized logits and computes the equivalent of log-softmax followed by negative log-likelihood [2]. We therefore do not apply softmax before `F.cross_entropy`.

Passing probabilities instead would make the loss treat those probabilities as a new set of logits, producing a different objective. If log-probabilities are needed separately, `F.log_softmax` avoids explicitly taking the logarithm of potentially tiny softmax outputs.

## 7. How the output head sends gradients backward

For one position, let $\mathbf{y}$ be a one-hot target vector and $\mathbf{p}$ the predicted probabilities. Differentiating cross-entropy gives

$$
\frac{\partial\ell}{\partial\mathbf{z}}=\mathbf{p}-\mathbf{y}.
$$

For our example with target index 3, this is approximately

$$
[0.0317,0.0861,0.2341,-0.3636,0.0117]^{\mathsf{T}}.
$$

The target coordinate has a negative derivative; the others have positive derivatives. Direct gradient descent on independent logits would increase the target score and decrease the others. In the network, parameter updates act through shared weights and representations, so their effects across examples are coupled.

Define $\mathbf{g}=\mathbf{p}-\mathbf{y}$. From $\mathbf{z}=W_{\mathrm{out}}\mathbf{h}+\mathbf{b}_{\mathrm{out}}$, the chain rule gives

$$
\frac{\partial\ell}{\partial W_{\mathrm{out}}}
=\mathbf{g}\mathbf{h}^{\mathsf{T}},
\qquad
\frac{\partial\ell}{\partial\mathbf{b}_{\mathrm{out}}}=\mathbf{g},
$$

$$
\frac{\partial\ell}{\partial\mathbf{h}}
=W_{\mathrm{out}}^{\mathsf{T}}\mathbf{g}.
$$

For the mean training loss, each position contributes a factor of $1/N$. Gradients accumulate across positions, pass through final LayerNorm, and continue through the decoder blocks to the embeddings. The vocabulary projection therefore connects the prediction objective to every earlier trainable component.

## 8. Reproducing the example in PyTorch

```python
import torch
from torch import nn
from torch.nn import functional as F

head = nn.Linear(4, 5).double()
with torch.no_grad():
    head.weight.zero_()
    head.bias.zero_()
    head.weight[:, 0].copy_(
        torch.tensor([0., 1., 2., 3., -1.], dtype=torch.float64)
    )

h = torch.tensor([[1., 0., 0., 0.]], dtype=torch.float64)
logits = head(h)
logits.retain_grad()
target = torch.tensor([3], dtype=torch.long)
probabilities = torch.softmax(logits, dim=-1)
loss = F.cross_entropy(logits, target)
loss.backward()

expected = probabilities.detach().clone()
expected[0, 3] -= 1.0

print(logits.detach())                 # [[0., 1., 2., 3., -1.]]
print(probabilities.detach())
print(loss.item())                     # Approximately 0.4519
print(torch.allclose(logits.grad, expected))  # True
print(logits.argmax(dim=-1))           # tensor([3])
```

`retain_grad()` makes the gradient of the intermediate logits available for inspection. It is not needed for ordinary training of the head's parameters.

## 9. From predictions to generation

Our generation method uses the last position's logits:

```python
logits = self(context)
next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
token_ids = torch.cat((token_ids, next_token), dim=1)
```

The last position summarizes the available prefix and predicts the next token to append. Earlier positions' predictions correspond to earlier prefixes, so they are not used for this next generation step.

Greedy decoding selects the largest logit. Softmax preserves score ordering, so calculating probabilities first would produce the same maximizing token. Greedy choice is deterministic for fixed logits, but it does not guarantee the most probable complete sequence.

After appending a token, the model runs again on the expanded context. This is different from teacher-forced training: newly generated tokens become subsequent inputs. Our implementation recomputes the context rather than caching attention keys and values.

The script calls `model.eval()` before generation, and the generation method uses `torch.no_grad()`. These serve different purposes: evaluation mode changes components such as dropout, while `no_grad()` disables gradient recording.

Two limits of this tiny demonstration are worth making explicit. First, `<eos>` is in the vocabulary but never appears as a target in the supplied training example. The model is therefore not being taught to end the sentence. Second, generation appends a fixed number of tokens; it has no special stopping rule for `<eos>`. For longer contexts, it retains the latest `max_len` tokens and assigns positions within that retained window.

A probability distribution is the model's prediction, not a guarantee of correctness or calibration. Training on one short sequence demonstrates the computation; it does not establish general language ability.

## References and connections

- [1] [PyTorch: softmax](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.softmax.html).
- [2] [PyTorch: cross-entropy loss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html).
- [Previous chapter: feed-forward network](https://github.com/emiliano-riva/transformer-decoder-only/blob/main/docs/modules/07-feed-forward-network.md).
- [Implementation: `TinyDecoderTransformer` and training loop](https://github.com/emiliano-riva/transformer-decoder-only/blob/main/src/tiny_transformer.py).
- [Statistical foundations: from logits to softmax](https://github.com/emiliano-riva/transformer-decoder-only/blob/main/docs/deep-dives/from-logits-to-softmax.md). The current introductory draft connects categorical models, log-odds, and cross-entropy; the full GLM tutorial remains to be integrated.
