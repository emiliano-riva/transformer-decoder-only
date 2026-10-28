# Token embeddings: from token IDs to learned vectors

Our Transformer receives a sequence of integers and produces predictions about the next token. The first step inside the network converts each integer into a vector of learnable values. This chapter explains that conversion, its relationship to matrix multiplication, and how the vectors participate in learning.

Prerequisites: vectors, matrix multiplication, and the idea that training adjusts model parameters to reduce a loss.

## 1. What a token ID represents

Our laboratory uses a small, manually defined vocabulary:

| Token | ID |
| --- | ---: |
| `il` | 0 |
| `gatto` | 1 |
| `mangia` | 2 |
| `pesce` | 3 |
| `<eos>` | 4 |

The input `il gatto mangia` becomes `[0, 1, 2]`. These integers identify entries in the vocabulary. Their numerical distances carry no intended linguistic meaning: ID 2 is not twice the meaning of ID 1, and adjacent IDs need not represent related tokens.

We could assign different IDs while preserving the model's computation, provided we consistently permuted the input embedding rows, output categories, and encoded data. What matters is the consistent association between each token and its parameters.

In this example, most tokens are whole words. More generally, a tokenizer can divide text into other units, including word fragments. Our script starts with an explicit vocabulary and already encoded sequences; tokenization itself is outside this chapter.

## 2. A table of learnable vectors

Let $V$ be the vocabulary size and $d$ the embedding dimension, which equals $d_{\text{model}}$ in our implementation. We define an embedding matrix

$$
E\in\mathbb{R}^{V\times d}.
$$

Row $i$ contains the vector associated with token ID $i$:

$$
\mathbf{e}_i=E_{i,:}\in\mathbb{R}^{d}.
$$

We use zero-based token IDs, so $i\in\{0,\ldots,V-1\}$. For $V=5$ and $d=4$, the matrix contains $5\times4=20$ learnable scalar parameters.

The dimension $d$ is a design choice. It is independent of the token's ID and the sequence length. Every token receives a vector of the same size, allowing subsequent layers to process all positions using shared transformations.

The coordinates are learned jointly with the rest of the model. We do not assign one coordinate to a predetermined property such as “animal” or “action.” Interpretable relationships may emerge through training, but individual coordinates need not have simple linguistic labels.

## 3. Following one sequence through the lookup

Consider this illustrative matrix. Its values are chosen for explanation; they are not learned results from our model:

$$
E=
\begin{bmatrix}
0.2 & -0.1 & 0.4 & 0.0\\
0.7 & 0.3 & -0.2 & 0.5\\
-0.4 & 0.6 & 0.1 & 0.8\\
0.5 & -0.3 & 0.9 & -0.2\\
0.0 & 0.2 & -0.5 & 0.1
\end{bmatrix}.
$$

The sequence `[0, 1, 2]` selects the first three rows:

$$
X_{\text{token}}=
\begin{bmatrix}
\mathbf{e}_0\\
\mathbf{e}_1\\
\mathbf{e}_2
\end{bmatrix}
=
\begin{bmatrix}
0.2 & -0.1 & 0.4 & 0.0\\
0.7 & 0.3 & -0.2 & 0.5\\
-0.4 & 0.6 & 0.1 & 0.8
\end{bmatrix}.
$$

For a batch of $B$ sequences of length $T$, lookup transforms the input shape as follows:

$$
(B,T)\longrightarrow(B,T,d).
$$

In our experiment, $(1,3)$ becomes $(1,3,4)$. The sequence still has three positions; each position now contains four real-valued features.

If a token appears twice, both occurrences initially retrieve the same row. Later operations can produce different representations for those occurrences because their positions and accessible contexts differ.

## 4. The connection to one-hot vectors

A token ID can also be represented by a one-hot row vector $\mathbf{o}_i\in\mathbb{R}^{1\times V}$, with a single 1 at index $i$. For `gatto`, whose ID is 1:

$$
\mathbf{o}_1=\begin{bmatrix}0&1&0&0&0\end{bmatrix}.
$$

Multiplication by $E$ selects precisely that row:

$$
\mathbf{o}_1E=E_{1,:}=\mathbf{e}_1.
$$

For one sequence, stacking the one-hot vectors into $O\in\mathbb{R}^{T\times V}$ gives

$$
X_{\text{token}}=OE.
$$

This establishes a connection with a linear transformation: an embedding lookup is mathematically equivalent to applying a shared matrix to one-hot inputs. The implementation can select rows directly, without constructing those mostly zero vectors.

One-hot vectors already distinguish token identities. Learning $E$ supplies a trainable representation that subsequent layers can use to reduce the prediction loss. Choosing $d<V$ also reduces the representation dimension, although this inequality is not a requirement of an embedding layer.

## 5. Where this appears in our implementation

In `TinyDecoderTransformer.__init__`, the layer is created with:

```python
self.token_embedding = nn.Embedding(vocab_size, d_model)
```

Its weight matrix has shape `(vocab_size, d_model)`. In the forward pass, `self.token_embedding(token_ids)` retrieves the vectors for all input positions. These shape conventions follow the PyTorch `Embedding` API [1].

The following standalone example reproduces our illustrative lookup:

```python
import torch
from torch import nn

embedding = nn.Embedding(5, 4)
example_weights = torch.tensor([
    [ 0.2, -0.1,  0.4,  0.0],
    [ 0.7,  0.3, -0.2,  0.5],
    [-0.4,  0.6,  0.1,  0.8],
    [ 0.5, -0.3,  0.9, -0.2],
    [ 0.0,  0.2, -0.5,  0.1],
])

with torch.no_grad():
    embedding.weight.copy_(example_weights)

token_ids = torch.tensor([[0, 1, 2]], dtype=torch.long)
vectors = embedding(token_ids)
print(vectors.shape)  # torch.Size([1, 3, 4])
print(vectors)
```

The manual assignment makes the example easy to inspect. In the training script, the embedding parameters begin with their default random initialization and are adjusted during optimization.

## 6. How the vectors learn

The training pair is:

```text
Input:   il     gatto    mangia
Target:  gatto  mangia   pesce
```

The selected embedding vectors influence the Transformer computations, the output logits, and ultimately the next-token prediction loss. Backpropagation follows that computational path to determine how changing the vectors would change the loss.

Write $I_{b,t}$ for the input token ID at batch index $b$ and position $t$, and let

$$
X_{b,t,j}=E_{I_{b,t},j}.
$$

If $G_{b,t,j}=\partial\mathcal{L}/\partial X_{b,t,j}$ is the gradient arriving at the lookup output, then

$$
\frac{\partial\mathcal{L}}{\partial E_{i,j}}
=\sum_{b,t:I_{b,t}=i}G_{b,t,j}.
$$

All occurrences of a token contribute to its shared embedding row. The incoming gradients already account for the later computations, including the ways a position can influence predictions at subsequent positions through causal attention.

For plain gradient descent, an illustrative update would be

$$
E\leftarrow E-\eta\nabla_E\mathcal{L}.
$$

Our script uses AdamW instead; it uses gradient history to determine the parameter updates. The gradient relationship above still describes the embedding lookup's contribution to backpropagation.

There is a useful consequence in this particular experiment: the input only contains IDs 0, 1, and 2. Rows 3 and 4 receive no gradient through lookup. Because our input embedding is separate from the output projection, those rows are never selected during training, and weight decay is zero, they remain at their initialization. The model can nevertheless learn to predict `pesce`: its output category has separate parameters in `lm_head`.

This distinction matters when interpreting the experiment. Memorizing one sequence does not demonstrate a useful semantic embedding space.

## 7. From token identity to position and context

At lookup time, the embedding depends only on the token ID. Our forward pass then adds a learned positional vector:

```python
x = self.token_embedding(token_ids) + self.position_embedding(positions)[None, :, :]
```

At position $t$, this gives

$$
\mathbf{x}^{(0)}_t=\mathbf{e}_{I_t}+\mathbf{p}_t.
$$

The sequence order is already preserved by the arrangement of the input tensor. Positional embeddings provide an explicit position-dependent signal in the representation passed to the decoder blocks. Causal masking additionally restricts which positions can communicate.

As the representation passes through those blocks, it becomes dependent on the accessible context. We therefore distinguish the token's shared lookup vector from its later hidden representation at a particular position.

The next chapter examines how we construct $\mathbf{p}_t$, why we add it to the token embedding, and what learned absolute positions imply for the model's context window.

## References and connections

- [1] [PyTorch: `nn.Embedding`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html), for API behavior and tensor shapes.
- [Implementation: `TinyDecoderTransformer`](https://github.com/emiliano-riva/transformer-decoder-only/blob/main/src/tiny_transformer.py).
- [Next chapter: positional embeddings](https://github.com/emiliano-riva/transformer-decoder-only/blob/main/docs/modules/02-positional-embedding.md).
- [Training and teacher forcing](https://github.com/emiliano-riva/transformer-decoder-only/blob/main/docs/02-training.md).
