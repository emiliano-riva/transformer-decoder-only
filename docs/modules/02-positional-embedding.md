# Positional embeddings: giving token representations a position

The previous chapter introduced a matrix that maps each token ID to a learned vector. Every occurrence of a token retrieves the same vector from that matrix. Our decoder then adds another learned vector, selected by the token's position in the input sequence.

This chapter follows that operation from its motivation to its implementation. We will use the same vocabulary, sequence, and four-dimensional representations as before.

Prerequisites: token embedding lookup, elementwise vector addition, and tensor shapes. The discussion of gradients builds on the previous chapter.

## 1. Token identity and sequence position

Consider the input:

| Position | Token | Token ID |
| ---: | --- | ---: |
| 0 | `il` | 0 |
| 1 | `gatto` | 1 |
| 2 | `mangia` | 2 |

Here, token IDs happen to match positions. This coincidence can obscure an important distinction. If we rearrange the tokens, their IDs stay attached to their vocabulary entries:

| Position | Token | Token ID |
| ---: | --- | ---: |
| 0 | `mangia` | 2 |
| 1 | `il` | 0 |
| 2 | `gatto` | 1 |

The ID answers “which vocabulary entry is this?” The position answers “where does this occurrence appear in the current input?”

Token embedding lookup preserves the sequence's arrangement: rearranging IDs rearranges the retrieved vectors. However, the vector retrieved for `gatto` is the same whether it occurs at position 0, 1, or 2. We want to give subsequent layers a position-dependent signal alongside token identity.

## 2. Why attention benefits from a positional signal

Attention compares representations to determine how information should be combined across positions. If we consider ordinary, unmasked self-attention with shared projections and no positional information, reordering its input rows simply reorders its output rows in the same way. This property is called permutation equivariance.

Such an operation can compare token contents, but the row number itself is not supplied as a feature. A positional representation makes the input to attention depend explicitly on where each token occurs.

Our decoder also has a causal mask. That mask already introduces an ordering constraint: position $t$ can access only positions up to $t$. Consequently, the permutation-equivariance statement for unmasked attention should not be applied unchanged to our masked decoder.

The two mechanisms have complementary roles:

| Mechanism | Role in our decoder |
| --- | --- |
| Positional embedding | Adds a learned representation of the current position to each token vector |
| Causal mask | Restricts which positions each attention query can access |

We use both. Adding position vectors alone would not prevent attention from reading future tokens.

## 3. A learned table of absolute positions

Let $L$ be the maximum supported input length and $d=d_{\text{model}}$ the representation dimension. We define a position embedding matrix

$$
P\in\mathbb{R}^{L\times d}.
$$

Row $t$ provides the vector for position $t$:

$$
\mathbf{p}_t=P_{t,:},\qquad t\in\{0,\ldots,L-1\}.
$$

In our model, $L=16$ and $d=4$. The position table therefore contains

$$
16\times4=64
$$

learnable scalar parameters. These are separate from the 20 parameters in the token embedding table.

The positions are called absolute because each vector is indexed by a position within the current input, rather than by the distance between two tokens. Position 2 retrieves the same row regardless of which token occupies it. In our implementation, every forward pass starts indexing at zero.

The entries are initialized randomly and learned during training. A position index such as 2 selects a row; it is not simply added as the scalar 2 to every coordinate. Also, independently learned rows do not inherently guarantee that neighboring positions have similar vectors.

## 4. Combining identity and position

For a sequence of token IDs $I_0,\ldots,I_{T-1}$, with $T\le L$, the initial decoder representation is

$$
\mathbf{x}^{(0)}_t=\mathbf{e}_{I_t}+\mathbf{p}_t.
$$

The superscript $(0)$ denotes the representation before the decoder blocks. Both terms have dimension $d$, so the addition preserves that dimension.

Reuse the illustrative token vectors from the previous chapter:

$$
X_{\text{token}}=
\begin{bmatrix}
0.2 & -0.1 & 0.4 & 0.0\\
0.7 & 0.3 & -0.2 & 0.5\\
-0.4 & 0.6 & 0.1 & 0.8
\end{bmatrix}.
$$

Choose the following illustrative vectors for positions 0, 1, and 2:

$$
P_{0:3,:}=
\begin{bmatrix}
0.1 & 0.0 & -0.1 & 0.2\\
-0.2 & 0.1 & 0.0 & 0.1\\
0.0 & -0.1 & 0.2 & -0.2
\end{bmatrix}.
$$

The sum is

$$
X^{(0)}=
\begin{bmatrix}
0.3 & -0.1 & 0.3 & 0.2\\
0.5 & 0.4 & -0.2 & 0.6\\
-0.4 & 0.5 & 0.3 & 0.6
\end{bmatrix}.
$$

These values are chosen for explanation and are not outputs of our training run.

For two occurrences of the same token at positions $s$ and $t$, their initial representations differ by

$$
\mathbf{x}^{(0)}_s-\mathbf{x}^{(0)}_t=\mathbf{p}_s-\mathbf{p}_t.
$$

Distinct learned position vectors can therefore distinguish the occurrences before attention mixes information from their contexts.

## 5. Why use addition?

Addition keeps each representation at width $d$. Our attention projections, normalization layers, and residual paths are built around this shared width. Concatenating two vectors of size $d$ would produce a vector of size $2d$, requiring a projection back to $d$ or changes to subsequent layers.

Addition allows both learned tables to influence the same downstream computation. For example, a linear map applied directly to the sum would satisfy

$$
(\mathbf{e}_{I_t}+\mathbf{p}_t)W
=\mathbf{e}_{I_t}W+\mathbf{p}_tW.
$$

This identity illustrates how both contributions can affect a projection. In our actual pre-norm decoder, LayerNorm intervenes before the attention projections, so the complete attention calculation is not described by this linear identity alone.

Addition does not guarantee that a later layer can recover the two original vectors independently. The model learns the two tables and downstream transformations jointly to support prediction. Other positional mechanisms are possible; this chapter focuses on the learned additive mechanism implemented in our script.

## 6. Reading the implementation and its shapes

The constructor creates the table:

```python
self.position_embedding = nn.Embedding(max_len, d_model)
```

The forward pass constructs the indices and combines the two lookups:

```python
positions = torch.arange(length, device=token_ids.device)
x = self.token_embedding(token_ids) + self.position_embedding(positions)[None, :, :]
```

For `length = 3`, `positions` contains `[0, 1, 2]`. These values are generated from the input length, independently of the token IDs.

| Expression | General shape | Our example |
| --- | --- | --- |
| `token_ids` | `(B, T)` | `(1, 3)` |
| `positions` | `(T,)` | `(3,)` |
| `self.token_embedding(token_ids)` | `(B, T, d)` | `(1, 3, 4)` |
| `self.position_embedding(positions)` | `(T, d)` | `(3, 4)` |
| Position vectors after `[None, :, :]` | `(1, T, d)` | `(1, 3, 4)` |
| Sum | `(B, T, d)` | `(1, 3, 4)` |

The `None` inserts a batch axis of size one. Broadcasting then applies the same position vectors to every sequence in the batch. This shares the position table across examples; it does not create a separate set of trainable position parameters for each example.

The following standalone calculation reproduces the numerical example:

```python
import torch

token_vectors = torch.tensor([[
    [ 0.2, -0.1,  0.4, 0.0],
    [ 0.7,  0.3, -0.2, 0.5],
    [-0.4,  0.6,  0.1, 0.8],
]])
position_vectors = torch.tensor([
    [ 0.1,  0.0, -0.1,  0.2],
    [-0.2,  0.1,  0.0,  0.1],
    [ 0.0, -0.1,  0.2, -0.2],
])

x = token_vectors + position_vectors[None, :, :]
print(x.shape)  # torch.Size([1, 3, 4])
print(x)
```

This example isolates the addition. The full model obtains both sets of vectors from its trainable embedding layers.

## 7. How position vectors learn

At coordinate $j$, the addition is

$$
X^{(0)}_{b,t,j}=E_{I_{b,t},j}+P_{t,j}.
$$

Let $G_{b,t,j}=\partial\mathcal{L}/\partial X^{(0)}_{b,t,j}$ be the incoming gradient. Since the same position row is used across the batch,

$$
\frac{\partial\mathcal{L}}{\partial P_{t,j}}
=\sum_b G_{b,t,j}.
$$

The derivative of addition with respect to either input is one. Consequently, the gradient at each occurrence flows into both the selected token row and the selected position row. Contributions accumulate by token identity in $E$ and by position in $P$.

Our training inputs always have length three. Only position rows 0, 1, and 2 are selected and receive training gradients. Rows 3 through 15 remain at initialization in this experiment, because they are never selected and weight decay is zero.

There is also a limitation in the data: `il` always appears at position 0, `gatto` at position 1, and `mangia` at position 2. This experiment does not establish that the model has learned to use those tokens at other positions or has acquired general rules about word order. A broader experiment would vary both sequences and lengths.

## 8. Position indices during generation

Our demonstration generates three tokens from the prefix `il`:

| Forward pass | Context, assuming the expected predictions | Position indices |
| ---: | --- | --- |
| 1 | `il` | `[0]` |
| 2 | `il gatto` | `[0, 1]` |
| 3 | `il gatto mangia` | `[0, 1, 2]` |

The final token `pesce` is appended after the third pass. Producing this four-token result therefore does not require looking up position 3. An additional generation step would process the four-token context and use that previously untrained position row.

The table supports input lengths up to 16, but supporting a shape is different from having learned useful behavior at every supported position.

For longer generated sequences, `generate()` retains only the latest `max_len` tokens:

```python
context = token_ids[:, -self.max_len:]
```

The next forward pass assigns positions starting from zero again. Once cropping occurs, these indices describe positions inside the retained window, not persistent positions in the entire generated sequence. A retained token can therefore receive a different position vector as the window shifts.

This is the behavior of our simple implementation. Increasing `max_len` or cropping a prefix does not by itself teach the model to handle longer contexts.

## 9. Entering the decoder blocks

After the embedding sum, the code executes:

```python
for block in self.blocks:
    x = block(x)
```

The positional vectors are added once, before the first block. Their influence continues through the subsequent transformations and residual connections; the table is not looked up and added again at each block.

We now have a sequence of representations that combine token identity with an explicit position signal. The next chapter examines causal self-attention: how each position uses these representations to gather information from its allowed context.

## References and connections

- [Previous chapter: token embeddings](https://github.com/emiliano-riva/transformer-decoder-only/blob/main/docs/modules/01-token-embedding.md).
- [Implementation: `TinyDecoderTransformer`](https://github.com/emiliano-riva/transformer-decoder-only/blob/main/src/tiny_transformer.py), especially `__init__`, `forward`, and `generate`.
- [PyTorch: `nn.Embedding`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html), for the lookup table API and shapes.
- [Next chapter: causal self-attention](https://github.com/emiliano-riva/transformer-decoder-only/blob/main/docs/modules/03-causal-self-attention.md).
