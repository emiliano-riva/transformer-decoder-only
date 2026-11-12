# 6. Softmax in Neural Networks

In the previous chapters, we derived the softmax function from the categorical distribution, represented as a member of the multiparameter exponential family.

We showed that, by choosing the multinomial canonical link, category probabilities can be expressed in terms of a vector of logits:

$$
\boxed{
p_j=\mathrm{softmax}(\mathbf z)_j
=\frac{e^{z_j}}{\sum_{h=1}^{K}e^{z_h}}
}
$$

This relationship provides the probabilistic foundation for numerous classification models.

In modern neural networks, however, softmax is not used exclusively to produce class probabilities. The same mathematical transformation also appears in Transformer *attention* mechanisms, where it serves a different purpose.

In this chapter, we examine both applications before exploring the connections among the categorical distribution, maximum likelihood, cross-entropy, and gradient computation.

## 6.1 From Multinomial Logistic Regression to Neural Networks

In multinomial logistic regression, logits are obtained from linear combinations of explanatory variables.

For an arbitrary observation $x$, we can write, with the intercept included in the covariate vector:

$$
z_j=x^\top\beta_j
$$

Alternatively, by separating the intercepts explicitly from the coefficients of the other covariates, we can use a matrix representation:

$$
\boxed{
\mathbf z=Wx+b
}
$$

where $W\in\mathbb R^{K\times d}$ is the coefficient matrix, $x\in\mathbb R^d$ contains the covariates without a constant component, and $b\in\mathbb R^K$ is the vector of intercepts. The two notations are equivalent provided the intercept is included only once.

With this formulation using $K$ logits, the parameters are not identifiable without additional constraints: adding the same predictor to every logit leaves the probabilities unchanged. In multinomial logistic regression with a reference category, the coefficients and intercept of that category can be fixed at zero.

Applying softmax yields the probabilities of the $K$ categories:

$$
P(Y=j\mid x)=
\frac{e^{z_j}}{\sum_{h=1}^{K}e^{z_h}}
$$

We can generalize this construction in neural networks.

Instead of computing the logits directly through a linear transformation of the input, we can use a more complex parametric function:

$$
\boxed{
\mathbf z=f_\psi(x)
}
$$

where $\psi$ denotes the set of neural network parameters.

The function $f_\psi$ may include multiple transformation layers, activation functions, residual connections, and attention mechanisms.

The final probabilities are still obtained through:

$$
\boxed{P(Y=j\mid x,\psi)=\mathrm{softmax}(f_\psi(x))_j}
$$

The fundamental difference from classical multinomial logistic regression therefore lies in how the logits are parameterized.

In a multinomial GLM, the log probability ratios relative to the reference category are linear in the covariates. In a neural network, they can instead depend on the input through a nonlinear function.

The overall model is not necessarily a GLM, but we can still use the categorical distribution to describe the response probabilistically.

This distinction allows us to retain the statistical foundations of softmax even when much more complex neural architectures are introduced.

## 6.2 Softmax at the Output of Language Models

One particularly important application of softmax is in autoregressive language models.

These models aim to predict the next token from the previously observed tokens.

Consider a sequence:

$$
(w_1,w_2,\ldots,w_T)
$$

where each $w_t$ is a token belonging to a vocabulary $\mathcal V$.

Let:

$$
V=|\mathcal V|
$$

denote the vocabulary size.

An autoregressive model defines a conditional distribution for each token in the sequence.

To predict $w_t$:

$$
P(w_t\mid w_1,\ldots,w_{t-1})
$$

which we can write more concisely as:

$$
P(w_t\mid w_{\lt t})
$$

The joint probability of the entire sequence can be factorized using the chain rule of probability:

$$
\boxed{P(w_1,\ldots,w_T)=\prod_{t=1}^{T}P(w_t\mid w_{\lt t})}
$$

For the first token, we assume an initial context, possibly represented by a special *beginning-of-sequence* (BOS) token. If there is no explicit initial token, the first factor is interpreted as the probability of the first token given an empty context. Termination can also be modeled by including a special end-of-sequence token.

### From Internal Representations to Logits

Suppose the network produces an internal representation at a particular sequence position:

$$
h_t\in\mathbb R^{d_{\mathrm{model}}}
$$

This representation contains the information the model has extracted from the available context.

In a decoder-only Transformer, it is produced by a sequence of blocks involving causal self-attention, feed-forward networks, normalization, and residual connections.

To produce a prediction over the vocabulary, we apply a linear projection:

$$
\boxed{
\mathbf z_t=W_{\mathrm{out}}h_t+b_{\mathrm{out}}
}
$$

where:

$$
W_{\mathrm{out}}\in\mathbb R^{V\times d_{\mathrm{model}}}
$$

The result is a vector of $V$ logits:

$$
\mathbf z_t\in\mathbb R^V
$$

Each component $z_{tj}$ represents the score assigned to vocabulary token $j$.

Applying softmax:

$$
\boxed{
p_{tj}=
\frac{e^{z_{tj}}}
{\sum_{h=1}^{V}e^{z_{th}}}
}
$$

we obtain a probability distribution over the vocabulary for a given token sequence.

Under the convention adopted in our decoder-only Transformer, the representation $h_t$ uses the tokens up to and including position $t$. Consequently, the logits at position $t$ predict the **next** token, at position $t+1$:

$$
\boxed{p_{tj}=P(w_{t+1}=j\mid w_{\leq t},\psi)}
$$

### Example: Next-Token Prediction

Consider the sequence:

> il gatto mangia pesce

During autoregressive training, the model may receive:

`il gatto mangia`

and use the following as targets:

`gatto mangia pesce`

Predictions for different positions are computed in parallel, but each position can use only the context allowed by the causal mask. In particular, the logits computed at position $t$ are compared with the target at position $t+1$: inputs and targets are shifted by one token.

For example:

| Available context | Token to predict |
|---|---|
| il | gatto |
| il gatto | mangia |
| il gatto mangia | pesce |

Each position has an associated probability distribution over the entire vocabulary.

During training, a technique known as *teacher forcing* is commonly used: the model receives the correct training-sequence tokens as context instead of recursively feeding back its own predictions.

Softmax thus transforms the logits produced at each position into the probabilities used to define the likelihood of the observed tokens.

## 6.3 Softmax in Self-Attention Mechanisms

In Transformers, softmax also appears inside the attention mechanism.

Its role, however, differs from the one described above.

In classification and language modeling, softmax is used to represent conditional probabilities over the possible categories of the response variable.

In self-attention, by contrast, it normalizes compatibility scores between representations of the elements in a sequence.

### Computing Attention Scores

Consider a sequence of $T$ representations collected in the matrix:

$$
X\in\mathbb R^{T\times d_{\mathrm{model}}}
$$

Three linear transformations produce:

$$
Q=XW_Q
$$

$$
K=XW_K
$$

$$
V_{\mathrm{att}}=XW_V
$$

The matrices $Q$, $K$, and $V_{\mathrm{att}}$ represent *Query*, *Key*, and *Value*, respectively.

For a single attention head, assume:

$$
Q,K\in\mathbb R^{T\times d_k}
$$

and:

$$
V_{\mathrm{att}}\in\mathbb R^{T\times d_v}
$$

The compatibility between the query at position $i$ and the key at position $j$ is computed using a dot product:

$$
s_{ij}=\frac{q_i^\top k_j}{\sqrt{d_k}}
$$

The scaling factor $1/\sqrt{d_k}$ limits the growth of dot-product magnitudes as dimensionality increases, helping prevent softmax from operating too frequently in saturated regions.

Collecting the scores into a matrix:

$$
S=\frac{QK^\top}{\sqrt{d_k}}
$$

### Softmax Normalization

For each position $i$, we apply softmax to the scores associated with positions $j$:

$$
\boxed{
a_{ij}=
\frac{e^{s_{ij}}}
{\sum_{h=1}^{T}e^{s_{ih}}}
}
$$

We obtain:

$$
a_{ij}>0
$$

and:

$$
\sum_{j=1}^{T}a_{ij}=1
$$

The values $a_{ij}$ are the **attention weights**.

For each query, these weights determine how much each Value vector contributes to the resulting representation. Without a mask, all weights in the formulas are strictly positive; we will shortly introduce causal masking, where some positions receive exactly zero weight:

$$
\boxed{
c_i=\sum_{j=1}^{T}a_{ij}v_j
}
$$

In matrix form:

$$
\boxed{
C=AV_{\mathrm{att}}
}
$$

where $A$ is the attention-weight matrix.

### The Causal Mask

In an autoregressive decoder-only Transformer, each position must be able to access only the information available up to that point in the sequence.

We therefore introduce a causal mask $M$, defined by:

$$
M_{ij}=
\begin{cases}
0 & \text{if }j\leq i\\
-\infty & \text{if }j>i
\end{cases}
$$

The weight matrix is computed as:

$$
\boxed{
A=
\mathrm{softmax}_{\mathrm{rows}}
\left(
\frac{QK^\top}{\sqrt{d_k}}+M
\right)
}
$$

Since:

$$
e^{-\infty}=0
$$

future positions receive zero weight. Thus, under causal masking, the statement $a_{ij}>0$ holds only for **allowed** positions ($j\leq i$); forbidden positions ($j>i$) have $a_{ij}=0$. The weights over allowed positions still sum to $1$.

Normalization is performed separately for each row, that is, for each query, considering only the positions permitted by the mask. This definition assumes that every query has at least one valid position, as is the case for a standard causal mask that includes the diagonal.

The mechanism is repeated across the different heads of Multi-Head Attention, each with its own projections.

### Two Different Uses of the Same Function

We can now clearly distinguish the two roles of softmax in Transformers.

| Feature | Output softmax | Attention softmax |
|---|---|---|
| Input | Class or token logits | Query-Key compatibility scores |
| Normalized dimension | Vocabulary categories | Sequence positions |
| Result | Predictive probabilities | Attention weights |
| Purpose | Model the response | Aggregate representations |
| Statistical connection | Categorical distribution of the output | Exponential normalization of scores |

Attention weights form a normalized distribution over the positions considered, but they do not automatically represent probabilities over the categories of the response variable.

In particular, they should not necessarily be interpreted as a direct measure of the causal importance of tokens to the final prediction.

The same mathematical transformation therefore serves two different operations: constructing a predictive distribution and determining aggregation weights.

---

## 6.4 From the Categorical Distribution to Maximum Likelihood

Let us return to multiclass classification.

In Chapter 4, we derived the probability mass function of the categorical distribution:

$$
P(\mathbf Y=\mathbf y)=\prod_{j=1}^{K}p_j^{y_j}
$$

where $\mathbf y$ denotes a one-hot encoding.

For a single observation, the model **likelihood** is:

$$
L(\psi;\mathbf y,x)=\prod_{j=1}^{K}p_j(x;\psi)^{y_j}
$$

The parameter $\psi$ can represent the coefficients of multinomial logistic regression or, more generally, the parameters of a neural network.

Taking the natural logarithm:

$$
\ell(\psi)=\ln L(\psi;\mathbf y,x)
$$

We obtain:

$$
\boxed{
\ell(\psi)=
\sum_{j=1}^{K}y_j\ln p_j
}
$$

This is the log-likelihood of a single categorical observation.

### Substituting Softmax

The probabilities are:

$$
p_j=
\frac{e^{z_j}}{\sum_{h=1}^{K}e^{z_h}}
$$

Substituting:

$$
\ell=
\sum_{j=1}^{K}
y_j\ln\left(
\frac{e^{z_j}}{\sum_{h=1}^{K}e^{z_h}}
\right)
$$

Using the identity:

$$
\ln\left(\frac ab\right)=\ln a-\ln b
$$

We obtain:

$$
\ell=
\sum_{j=1}^{K}
y_j\left[z_j-\ln\left(\sum_{h=1}^{K}e^{z_h}\right)\right]
$$

Distributing the summation:

$$
\ell=\sum_{j=1}^{K}y_jz_j-\sum_{j=1}^{K}y_j\ln\left(\sum_{h=1}^{K}e^{z_h}\right)
$$

Since the one-hot representation satisfies:

$$
\sum_{j=1}^{K}y_j=1
$$

we obtain:

$$
\boxed{\ell=\sum_{j=1}^{K}y_jz_j-\ln\left(\sum_{h=1}^{K}e^{z_h}\right)}
$$

Maximum likelihood estimation consists of finding the parameters that maximize this expression over the available observations.

## 6.5 From Maximum Likelihood to Cross-Entropy

Neural networks are commonly trained using optimization algorithms that minimize a *loss function*.

We can therefore transform the maximization of the log-likelihood into the equivalent problem of minimizing its negative.

Define:

$$
\mathcal L=-\ell
$$

We obtain:

$$
\boxed{\mathcal L=-\sum_{j=1}^{K}y_j\ln p_j}
$$

This expression is the **categorical cross-entropy** between the empirical distribution represented by the one-hot vector and the distribution predicted by the model.

For a single observation belonging to the correct class $c$, we have:

$$
y_c=1
$$

while:

$$
y_j=0,\qquad j\neq c
$$

The cross-entropy therefore reduces to:

$$
\boxed{
\mathcal L=-\ln p_c
}
$$

The model is penalized according to the probability it assigns to the category actually observed.

If that probability is high, the loss is small; if it is very low, the loss increases.

### Extension to a Sample

Consider $n$ observations.

Assuming conditional independence, their joint likelihood is the product of the individual likelihoods.

The mean negative log-likelihood becomes:

$$
\boxed{\mathcal L_{\mathrm{media}}=-\frac1n\sum_{i=1}^{n}\sum_{j=1}^{K}y_{ij}\ln p_{ij}}
$$

Therefore:

$$
\boxed{\arg\min_\psi\mathcal L_{\mathrm{media}}(\psi)=\arg\max_\psi\ell_{\mathrm{totale}}(\psi)}
$$

subject to the same constraints and in the absence of additional regularization terms, **provided the optima are actually attained**. Otherwise, the objective of maximizing the log-likelihood is still equivalent to minimizing its negative.

Cross-entropy is therefore not an arbitrarily chosen loss function: it follows directly from the likelihood of the categorical distribution.

### Cross-Entropy in Autoregressive Sequences

In language modeling, the same construction is applied at different positions in the sequence.

If the correct token at position $t$ is $w_t$, the corresponding loss is:

$$
\mathcal L_t=-\ln P(w_t\mid w_{\lt t},\psi)
$$

Summing over the positions considered:

$$
\boxed{
\mathcal L_{\mathrm{sequenza}}=-\sum_{t=1}^{T}\ln P(w_t\mid w_{\lt t},\psi)}
$$

This expression follows directly from the autoregressive factorization of the joint sequence probability. Here the loss is indexed by the **token to be predicted**, $w_t$, whereas in Section 6.2 the logits $\mathbf z_t$ are indexed by the **input position** that predicts $w_{t+1}$. The two conventions are consistent as long as the one-token shift is maintained.

In practice, during training, the loss is often averaged over the number of valid tokens in a batch, excluding padding positions where appropriate.

---

## 6.6 The Gradient of Cross-Entropy with Respect to Logits

One of the most important results of combining softmax with categorical cross-entropy is the simplicity of the gradient.

We start with the loss for a single observation, expressed directly in terms of the logits:

$$
\mathcal L=
-\sum_{j=1}^{K}y_jz_j+
\ln\left(\sum_{h=1}^{K}e^{z_h}\right)
$$

We want to compute the derivative with respect to an arbitrary logit $z_k$:

$$
\frac{\partial\mathcal L}{\partial z_k}
$$

### Derivative of the First Term

Consider:

$$
-\sum_{j=1}^{K}y_jz_j
$$

When differentiating with respect to $z_k$, only the term with index $j=k$ contributes:

$$
\frac{\partial}{\partial z_k}
\left(
-\sum_{j=1}^{K}y_jz_j
\right)
=-y_k
$$

### Derivative of the Second Term

Now consider:

$$
\ln\left(\sum_{h=1}^{K}e^{z_h}\right)
$$

Applying the chain rule:

$$
\begin{aligned}
&\frac{\partial}{\partial z_k}
\ln\left(\sum_{h=1}^{K}e^{z_h}\right)\\
&=
\frac{1}{\sum_{h=1}^{K}e^{z_h}}
\frac{\partial}{\partial z_k}
\left(\sum_{h=1}^{K}e^{z_h}\right)
\end{aligned}
$$

In the sum, only the term $e^{z_k}$ depends on $z_k$.

Therefore:

$$
\frac{\partial}{\partial z_k}
\left(\sum_{h=1}^{K}e^{z_h}\right)
=e^{z_k}
$$

We obtain:

$$
\frac{\partial}{\partial z_k}
\ln\left(\sum_{h=1}^{K}e^{z_h}\right)=
\frac{e^{z_k}}{\sum_{h=1}^{K}e^{z_h}}
$$

### Combining the Results

Combining the two contributions:

$$
\frac{\partial\mathcal L}{\partial z_k}=
-y_k+
\frac{e^{z_k}}{\sum_{h=1}^{K}e^{z_h}}
$$

We recognize the softmax function:

$$
p_k=
\frac{e^{z_k}}{\sum_{h=1}^{K}e^{z_h}}
$$

We therefore obtain:

$$
\boxed{
\frac{\partial\mathcal L}{\partial z_k}
=p_k-y_k
}
$$

In vector form:

$$
\boxed{
\nabla_{\mathbf z}\mathcal L=
\mathbf p-\mathbf y
}
$$

This result is particularly important because the gradient of the loss with respect to the logits equals the difference between the vector of predicted probabilities and the vector of observed outcomes.

The relationship follows from the mathematical structure of softmax combined with the categorical negative log-likelihood.

It is not a general property of every normalization function or every loss function.

### Connection to the Cumulant Function

In Chapter 4, we derived the cumulant function of the categorical distribution.

In the representation with $K$ logits, introduce the function:

$$
A(\mathbf z)=
\ln\left(\sum_{j=1}^{K}e^{z_j}\right)
$$

This is the *log-sum-exp* function. It is the logarithmic normalization function of the **redundant** representation with $K$ logits. The cumulant function of the *minimal* exponential family introduced in Chapter 4 is recovered by setting $z_K=0$ and $z_j=\theta_j$ for $j<K$:

$$
A(\boldsymbol\theta)=\ln\left(1+\sum_{j=1}^{K-1}e^{\theta_j}\right)
$$

Its derivative with respect to $z_k$ in the redundant representation gives:

$$
\frac{\partial A}{\partial z_k}=p_k
$$

The loss can be written as:

$$
\mathcal L=A(\mathbf z)-\mathbf y^\top\mathbf z
$$

Consequently:

$$
\nabla_{\mathbf z}\mathcal L=
\nabla A(\mathbf z)-\mathbf y
$$

We recover:

$$
\boxed{
\nabla_{\mathbf z}\mathcal L=\mathbf p-\mathbf y
}
$$

This is the direct connection between the exponential-family cumulant function and the cross-entropy gradient. With $K$ logits, the Hessian of $A(\mathbf z)$ is $\mathrm{diag}(\mathbf p)-\mathbf p\mathbf p^\top$: it is positive semidefinite but singular in the $\mathbf1$ direction, consistently with the translation invariance of softmax.

## 6.7 From the Gradient on Logits to the Gradient on Parameters

We have shown that:

$$
\frac{\partial\mathcal L_i}{\partial z_{ij}}
=p_{ij}-y_{ij}
$$

This derivative measures the sensitivity of the loss for observation $i$ to the logit of category $j$.

To update the model parameters, however, we must differentiate the loss with respect to the coefficients that determine the logits.

First, consider the linear case:

$$
z_{ij}=x_i^\top\beta_j
$$

where $\beta_j$ is the vector of coefficients associated with category $j$.

### Derivative of the Logit with Respect to the Coefficients

Expanding the dot product:

$$
z_{ij}=
\sum_{r=1}^{d}x_{ir}\beta_{jr}
$$

Differentiating with respect to an arbitrary component $\beta_{jr}$:

$$
\frac{\partial z_{ij}}{\partial\beta_{jr}}
=x_{ir}
$$

Collecting the derivatives with respect to all components of $\beta_j$:

$$
\boxed{
\nabla_{\beta_j}z_{ij}=x_i
}
$$

### Applying the Chain Rule

The loss depends on $\beta_j$ through the logits. Since $\beta_j$ is a **vector**, we use $\nabla_{\beta_j}$ to denote its vector gradient.

The multivariate chain rule gives:

```math
\nabla_{\beta_j}\mathcal L_i=\sum_{k=1}^{K}
\frac{\partial\mathcal L_i}{\partial z_{ik}}
\nabla_{\beta_j}z_{ik}
```

In the linear model considered here, each logit $z_{ik}$ depends only on the coefficient vector $\beta_k$.

Therefore:

$$
\nabla_{\beta_j}z_{ik}=
\begin{cases}
x_i & \text{if }k=j\\
0 & \text{if }k\neq j
\end{cases}
$$

Only the term with $k=j$ remains in the sum:

$$
\nabla_{\beta_j}\mathcal L_i=
\frac{\partial\mathcal L_i}{\partial z_{ij}}
\nabla_{\beta_j}z_{ij}
$$

Substituting:

$$
\frac{\partial\mathcal L_i}{\partial z_{ij}}
=p_{ij}-y_{ij}
$$

and:

$$
\nabla_{\beta_j}z_{ij}=x_i
$$

we obtain:

$$
\boxed{
\nabla_{\beta_j}\mathcal L_i=
(p_{ij}-y_{ij})x_i
}
$$

### Mean Gradient Over the Entire Sample

The mean loss over $n$ observations is:

$$
\mathcal L_{\mathrm{media}}=
\frac1n\sum_{i=1}^{n}\mathcal L_i
$$

Differentiating with respect to $\beta_j$:

$$
\nabla_{\beta_j}\mathcal L_{\mathrm{media}}=
\frac1n\sum_{i=1}^{n}
\nabla_{\beta_j}\mathcal L_i
$$

Substituting the previous result:

$$
\boxed{
\nabla_{\beta_j}\mathcal L_{\mathrm{media}}=
\frac1n\sum_{i=1}^{n}
(p_{ij}-y_{ij})x_i
}
$$

This is the expression for the mean categorical cross-entropy gradient with respect to the coefficients associated with category $j$. If the intercepts $b_j$ are parameterized separately from the other coefficients, then for a single observation we also have $\frac{\partial\mathcal L_i}{\partial b_j}=p_{ij}-y_{ij}$.

The derivation uses $K$ coefficient vectors in the symmetric, redundant softmax parameterization. If we select a reference category and fix its coefficients at zero, the same formula applies to the remaining $K-1$ free coefficient vectors.

### Extension to Neural Networks

In neural networks, the logits are generally nonlinear functions of the parameters:

$$
\mathbf z_i=f_\psi(x_i)
$$

Applying the chain rule:

$$
\boxed{
\nabla_\psi\mathcal L_i=
\sum_{j=1}^{K}
(p_{ij}-y_{ij})\nabla_\psi z_{ij}
}
$$

The principle remains the same: $p_{ij}-y_{ij}$ gives the gradient with respect to the logit, while the derivatives of the logits with respect to the parameters depend on the network architecture. If $J_{f_\psi}(x_i)$ is the Jacobian of the logits with respect to the parameters, the same rule can be written compactly as $\nabla_\psi\mathcal L_i=J_{f_\psi}(x_i)^\top(\mathbf p_i-\mathbf y_i)$.

Through **backpropagation**, this gradient is propagated backward through the operations that produced the logits.

An optimization algorithm, such as SGD or AdamW, then uses these gradients to update the parameters.

For classical gradient descent:

$$
\boxed{
\psi^{(t+1)}=
\psi^{(t)}-\alpha\nabla_\psi\mathcal L
}
$$

where $\alpha$ is the learning rate.

This relationship shows how maximum likelihood in classical statistics and gradient descent optimization share the same mathematical foundation when using a categorical distribution parameterized by softmax.

---

## 6.8 Numerical Example of the Gradient

Consider a classifier with three categories. This example is independent of the logits $(2,1,0)$ used in the previous chapter and of the PyTorch example in Section 6.9.

Suppose it outputs the probability vector:

$$
\mathbf p=(0.7,\ 0.2,\ 0.1)
$$

The correct class is the first one, so the one-hot vector is:

$$
\mathbf y=(1,\ 0,\ 0)
$$

The cross-entropy is:

$$
\mathcal L=-\ln(0.7)
$$

We obtain:

$$
\boxed{
\mathcal L\approx0.357
}
$$

The gradient with respect to the logits is:

$$
\nabla_{\mathbf z}\mathcal L=
\mathbf p-\mathbf y
$$

Substituting:

$$
\nabla_{\mathbf z}\mathcal L=
(0.7,\ 0.2,\ 0.1)-(1,\ 0,\ 0)
$$

Therefore:

$$
\boxed{
\nabla_{\mathbf z}\mathcal L=
(-0.3,\ 0.2,\ 0.1)
}
$$

How should we interpret these values?

For the first category, the gradient is negative:

$$
\frac{\partial\mathcal L}{\partial z_1}=-0.3
$$

Updating the logits directly in the direction opposite to the gradient would tend to increase $z_1$, raising the probability assigned to the correct class.

For the other categories, the gradients are positive:

$$
\frac{\partial\mathcal L}{\partial z_2}=0.2
$$

$$
\frac{\partial\mathcal L}{\partial z_3}=0.1
$$

The same direct update would tend to decrease their respective logits.

In a neural network, however, logits are not independent parameters that are directly updated: the gradient is propagated to the weights that produced them, so the effect of an actual parameter update also depends on the network structure.

Also note that:

$$
\sum_{j=1}^{K}(p_j-y_j)=0
$$

Indeed:

$$
\sum_{j=1}^{K}p_j=1
$$

and:

$$
\sum_{j=1}^{K}y_j=1
$$

This property is consistent with the invariance of softmax to adding a common constant to all logits: the loss does not change when all logits are shifted by the same amount.

## 6.9 PyTorch Implementation

In numerical implementations, it is important to distinguish between explicitly computing probabilities and computing the loss function.

PyTorch provides the function:

`torch.nn.functional.cross_entropy`

which accepts logits directly and computes cross-entropy using numerically stable operations.

There is therefore no need to apply softmax explicitly before passing the logits to the function.

### Example with Three Categories

```python
import torch
import torch.nn.functional as F

logits = torch.tensor(
    [[2.0, 1.0, 0.0]],
    requires_grad=True
)

target = torch.tensor([0])

loss = F.cross_entropy(logits, target)

loss.backward()

print("Probabilities:", torch.softmax(logits, dim=-1))
print("Loss:", loss.item())
print("Gradient:", logits.grad)
```

The `cross_entropy` function conceptually combines *log-softmax* with the negative log-likelihood.

The gradient computed automatically by PyTorch corresponds to the relationship (for a single observation with `mean` reduction, which introduces no additional scaling in this case):

$$
\nabla_{\mathbf z}\mathcal L=\mathbf p-\mathbf y
$$

for a single categorical observation with a one-hot target. In the code, target `0` is the index of the first class; `F.cross_entropy` accepts class indices directly and therefore does not require an explicit one-hot vector. With logits $(2,1,0)$, the loss is approximately $0.407606$ and the gradient is approximately $(-0.334759,\ 0.244728,\ 0.090031)$.

### Tensor Shapes in Language Modeling

Consider a batch of sequences with:

- $B$: number of sequences in the batch;
- $T$: sequence length;
- $V$: vocabulary size.

The model produces logits with shape:

$$
\boxed{
(B,T,V)
}
$$

The targets instead contain the indices of the correct tokens:

$$
\boxed{
(B,T)
}
$$

To use cross-entropy, we can reshape the tensors:

```python
B, T, V = logits.shape

loss = F.cross_entropy(
    logits.reshape(B * T, V),
    targets.reshape(B * T)
)
```

In this way, each position of each sequence is treated as a separate categorical prediction.

The function computes the mean loss over the $B\cdot T$ positions, assuming all positions are valid. If padding tokens are present, they must be appropriately excluded from the loss, for example by setting the targets at excluded positions to `-100`, the default `ignore_index` in `F.cross_entropy`. In the absence of other weights, the mean is then computed only over non-ignored positions.

This operation corresponds to minimizing the negative log-likelihood of the observed tokens, averaged over the positions considered.

During inference, however, when we need explicit probabilities for the tokens, we can compute them using:

```python
probabilities = torch.softmax(logits, dim=-1)
```

Normalization is performed along the last tensor dimension, corresponding to the $V$ categories in the vocabulary.

## 6.10 Final Remarks

We have shown how the softmax function, derived from multinomial logistic regression, is used in different components of neural networks.

In classifiers and autoregressive language models, softmax allows us to parameterize conditional categorical distributions.

In attention mechanisms, it instead normalizes compatibility scores to obtain the weights used to aggregate token representations.

We also derived categorical cross-entropy as the negative log-likelihood of the categorical distribution:

$$
\boxed{
\mathcal L=
-\sum_{j=1}^{K}y_j\ln p_j
}
$$

and showed that the gradient with respect to the logits takes the form:

$$
\boxed{
\nabla_{\mathbf z}\mathcal L=\mathbf p-\mathbf y
}
$$

This simple expression provides a fundamental link between probabilistic modeling of the response and neural network optimization through backpropagation.

The same statistical structure used to estimate coefficients in multinomial logistic regression can therefore also define the training objective of much more complex models, including autoregressive Transformers.

It is important, however, to distinguish the production of probabilities from their statistical reliability.

A classifier assigning a high probability to a category does not automatically guarantee that this probability is well calibrated, nor does it provide a complete measure of uncertainty about the model parameters.

These issues require additional tools for statistical inference and probabilistic evaluation.

In the next chapter, we will therefore examine the inferential possibilities offered by probabilistic models, distinguishing the frequentist approach, the Bayesian approach, and predictive uncertainty.
