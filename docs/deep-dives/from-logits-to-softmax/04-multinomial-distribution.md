# 4. Generalization to the Multinomial Distribution

In the previous chapter, we showed how the Bernoulli distribution can be represented as a member of the exponential family, deriving its natural parameter:

$$
\theta=\ln\left(\frac{p}{1-p}\right)
$$

Choosing the canonical link by imposing $\eta=\theta$ leads to the logit function and, through its inverse, to the logistic or sigmoid function.

This construction is particularly well suited to binary classification problems, in which the response variable can take only two values.

We now want to generalize this procedure to a response variable that can belong to one of $K$ mutually exclusive categories.

To do so, we will start from the **categorical distribution**, which is equivalent to a multinomial distribution with a single trial, and show that it too belongs to the exponential family.

We will also see how the probability normalization constraint leads to a representation with $K-1$ natural parameters and a multivariate cumulant function.

## 4.1 From the Bernoulli to the Categorical Distribution

Consider a random variable $Y$ that can take one of $K$ distinct values, with $K\geq 2$:

$$
Y\in\{1,2,\ldots,K\}
$$

Define the probability associated with a generic category $j$:

$$
p_j=P(Y=j)
$$

where:

$$
j=1,\ldots,K
$$

The probabilities must satisfy two conditions:

$$
p_j\geq0
$$

and:

$$
\boxed{
\sum_{j=1}^{K}p_j=1
}
$$

The second condition follows from the fact that the categories are mutually exclusive and exhaustive: each observation belongs to exactly one category.

In the special case $K=2$, the categorical distribution is equivalent to a Bernoulli distribution, apart from the encoding of the categories.

### One-Hot Representation

To represent category membership mathematically, we introduce a vector of indicator variables:

$$
\mathbf Y=(Y_1,Y_2,\ldots,Y_K)
$$

where:

$$
Y_j=\begin{cases}1 & \text{if }Y=j\\0 & \text{otherwise}\end{cases}
$$

This representation is called **one-hot encoding**.

For example, with three categories, the possible outcomes are:

| Observed category | $Y_1$ | $Y_2$ | $Y_3$ |
|---|---:|---:|---:|
| Class 1 | 1 | 0 | 0 |
| Class 2 | 0 | 1 | 0 |
| Class 3 | 0 | 0 | 1 |

In each case, exactly one component equals $1$, while all the others equal zero.

Consequently:

$$
\boxed{\sum_{j=1}^{K}Y_j=1}
$$

The distribution of the vector $\mathbf Y$ can be written as:

$$
\mathbf Y\sim\mathrm{Multinomial}(1,\mathbf p)
$$

where:

$$
\mathbf p=(p_1,p_2,\ldots,p_K)
$$

The first argument of the multinomial distribution specifies the number of trials, which in this case is one.

Therefore, the categorical distribution is the special case of the multinomial distribution with a single trial.

### Probability Mass Function

Using the one-hot representation, the probability mass function can be written as:

$$
\boxed{P(\mathbf Y=\mathbf y)=\prod_{j=1}^{K}p_j^{y_j}}
$$

Let us verify this expression.

Suppose we have three categories and observe the second one:

$$
\mathbf y=(0,1,0)
$$

Then:

$$
P(\mathbf Y=(0,1,0))=p_1^0p_2^1p_3^0
$$

and therefore:

$$
P(\mathbf Y=(0,1,0))=p_2
$$

The representation correctly returns the probability of the observed category.

For $n$ independent trials with the same category probabilities, the overall category counts instead follow the general multinomial distribution, whose probability mass function also includes the multinomial coefficient.

We will initially consider a single observation and then extend the model to a sample.

## 4.2 The Normalization Constraint and the $K-1$ Degrees of Freedom

Before deriving the exponential-family representation, we must examine an important consequence of the constraint:

$$
\sum_{j=1}^{K}Y_j=1
$$

From this relationship, we can isolate the last component:

$$
\boxed{
Y_K=1-\sum_{j=1}^{K-1}Y_j
}
$$

This means that, once the first $K-1$ components of the vector are known, the last one is uniquely determined.

For example, when $K=3$:

$$
Y_3=1-Y_1-Y_2
$$

The possible configurations are:

| $Y_1$ | $Y_2$ | $Y_3=1-Y_1-Y_2$ |
|---:|---:|---:|
| 1 | 0 | 0 |
| 0 | 1 | 0 |
| 0 | 0 | 1 |

Thus, $Y_1+Y_2$ equals $1$ if either of the first two categories is observed, and $0$ if the third category is observed.

### Free Coordinates and Statistical Independence

The constraint implies that only $K-1$ coordinates are needed to represent the indicator vector.

However, this does not mean that the first $K-1$ variables are statistically independent. For $K\geq 3$, there are at least two indicators among these components, and they are dependent when both corresponding categories have positive probabilities.

Indeed, if one indicator equals $1$, all the others must necessarily equal $0$.

For two distinct categories $j$ and $h$, with $j\neq h$:

$$
P(Y_j=1,Y_h=1)=0
$$

whereas, if both categories have positive probabilities:

$$
P(Y_j=1)P(Y_h=1)=p_jp_h>0
$$

Therefore, the indicators are statistically dependent.

The important distinction is that **the indicator vector is subject to a linear constraint and has only $K-1$ nonredundant coordinates**.

The same principle applies to the probability vector:

$$
\sum_{j=1}^{K}p_j=1
$$

Hence:

$$
\boxed{p_K=1-\sum_{j=1}^{K-1}p_j}
$$

Knowing the first $K-1$ probabilities determines the last one.

Nevertheless, the first $K-1$ probabilities cannot be chosen arbitrarily: they must be nonnegative, and their sum must not exceed $1$.

This structure introduces $K-1$ degrees of freedom into the parameterization of the distribution.

For the remainder of the derivation, we assume:

$$
p_j>0,\qquad j=1,\ldots,K
$$

so that all logarithms of probability ratios are well defined and finite. This assumption restricts our discussion to the **interior of the probability simplex**: cases with zero probabilities correspond to limiting values of the natural parameters, not finite values.

## 4.3 Exponential-Family Representation

Recall the probability mass function of the categorical distribution:

$$
P(\mathbf Y=\mathbf y)=\prod_{j=1}^{K}p_j^{y_j}
$$

Take the natural logarithm:

$$
\ln P(\mathbf Y=\mathbf y)=\ln\left(\prod_{j=1}^{K}p_j^{y_j}\right)
$$

Using the properties of logarithms:

$$
\boxed{
\ln P(\mathbf Y=\mathbf y)=\sum_{j=1}^{K}y_j\ln p_j}
$$

Our goal is to rewrite this expression in the form of a multiparameter exponential family.

### Introducing a Reference Category

Since only $K-1$ probabilities are free, we can choose one category as the reference and express the other probabilities relative to it.

By convention, we choose category $K$.

From the constraint on the one-hot representation:

$$
y_K=1-\sum_{j=1}^{K-1}y_j
$$

Separate the term corresponding to the reference category:

$$
\ln P(\mathbf Y=\mathbf y)=\sum_{j=1}^{K-1}y_j\ln p_j+y_K\ln p_K
$$

Substitute the expression for $y_K$:

$$
\begin{aligned}\ln P(\mathbf Y=\mathbf y)&=\sum_{j=1}^{K-1}y_j\ln p_j\\&\quad+\left(1-\sum_{j=1}^{K-1}y_j\right)\ln p_K\end{aligned}
$$

Expanding:

$$
\begin{aligned}\ln P(\mathbf Y=\mathbf y)&=\sum_{j=1}^{K-1}y_j\ln p_j+\ln p_K\\&\quad-\sum_{j=1}^{K-1}y_j\ln p_K\end{aligned}
$$

Collect the terms involving $y_j$:

$$
\ln P(\mathbf Y=\mathbf y)=\sum_{j=1}^{K-1}y_j(\ln p_j-\ln p_K)+\ln p_K
$$

Applying the logarithm-of-a-ratio identity:

$$
\boxed{\ln P(\mathbf Y=\mathbf y)=\sum_{j=1}^{K-1}y_j\ln\left(\frac{p_j}{p_K}\right)+\ln p_K}
$$

This expression is analogous to the one obtained for the Bernoulli distribution.

In the binary case, we identified the natural parameter:

$$
\theta=\ln\left(\frac{p}{1-p}\right)
$$

In the categorical case, we can instead identify one natural parameter for each of the first $K-1$ categories:

$$
\boxed{\theta_j=\ln\left(\frac{p_j}{p_K}\right),\qquad j=1,\ldots,K-1}
$$

We thus obtain a vector of natural parameters:

$$
\boxed{\boldsymbol\theta=\begin{pmatrix}\theta_1\\\theta_2\\\vdots\\\theta_{K-1}\end{pmatrix}}
$$

where:

$$
\boldsymbol\theta\in\mathbb R^{K-1}
$$

Each parameter $\theta_j$ represents the logarithm of the ratio between the probability of category $j$ and that of the reference category $K$. This is the **minimal reference-category parameterization**: choosing category $K$ is a convention, while the number of free natural coordinates remains $K-1$.

For $K>2$, unlike the Bernoulli distribution, the categorical distribution requires multiple natural parameters. When $K=2$, we recover the single Bernoulli parameter: setting $p_1=p$ and $p_2=1-p$ yields $\theta_1=\ln[p/(1-p)]$.

The next step is to derive the corresponding cumulant function.

## 4.4 Deriving the Multivariate Cumulant Function

Starting from the definition of the natural parameters:

$$
\theta_j=\ln\left(\frac{p_j}{p_K}\right)
$$

exponentiate both sides:

$$
e^{\theta_j}=\frac{p_j}{p_K}
$$

Multiplying by $p_K$:

$$
\boxed{p_j=p_Ke^{\theta_j}}
$$

This relationship allows us to express the probability of each category relative to that of the reference category.

### Using the Normalization Constraint

We know that:

$$
\sum_{j=1}^{K}p_j=1
$$

Separate the probability of category $K$:

$$
p_K+\sum_{j=1}^{K-1}p_j=1
$$

Substitute:

$$
p_j=p_Ke^{\theta_j}
$$

We obtain:

$$
p_K+\sum_{j=1}^{K-1}p_Ke^{\theta_j}=1
$$

Factor out $p_K$:

$$
p_K\left(1+\sum_{j=1}^{K-1}e^{\theta_j}\right)=1
$$

Hence:

$$
\boxed{p_K=\frac{1}{1+\sum_{j=1}^{K-1}e^{\theta_j}}}
$$

We have now expressed the probability of the reference category as a function of the $K-1$ natural parameters.

### Deriving the Other Probabilities

For the first $K-1$ categories, we previously obtained:

$$
p_j=p_Ke^{\theta_j}
$$

Substitute the expression just derived for $p_K$:

$$
p_j=\left(\frac{1}{1+\sum_{h=1}^{K-1}e^{\theta_h}}\right)e^{\theta_j}
$$

Moving the exponential factor to the numerator:

$$
\boxed{p_j=\frac{e^{\theta_j}}{1+\sum_{h=1}^{K-1}e^{\theta_h}},\qquad j=1,\ldots,K-1}
$$

The index $j$ identifies the category whose probability we want to calculate, while $h$ is the summation index used to include the contributions from all non-reference categories.

The probabilities therefore share the same denominator.

### Identifying the Cumulant Function

Recall the log-probability derived in the previous section:

$$
\ln P(\mathbf Y=\mathbf y)=\sum_{j=1}^{K-1}y_j\theta_j+\ln p_K
$$

Since:

$$
p_K=\frac{1}{1+\sum_{j=1}^{K-1}e^{\theta_j}}
$$

taking the logarithm gives:

$$
\ln p_K=-\ln\left(1+\sum_{j=1}^{K-1}e^{\theta_j}\right)
$$

Substituting:

$$
\begin{aligned}\ln P(\mathbf Y=\mathbf y)&=\sum_{j=1}^{K-1}y_j\theta_j\\&\quad-\ln\left(1+\sum_{j=1}^{K-1}e^{\theta_j}\right)\end{aligned}
$$

We identify the **multivariate cumulant function**:

$$
\boxed{A(\boldsymbol\theta)=\ln\left(1+\sum_{j=1}^{K-1}e^{\theta_j}\right)}
$$

The probability mass function can therefore be expressed as:

$$
\boxed{P(\mathbf Y=\mathbf y)=\exp\left[\sum_{j=1}^{K-1}y_j\theta_j-A(\boldsymbol\theta)\right]}
$$

Or, using the dot product:

$$
\boxed{P(\mathbf Y=\mathbf y)=\exp\left[\mathbf y_{1:K-1}^{\top}\boldsymbol\theta-A(\boldsymbol\theta)\right]}
$$

where $\mathbf y_{1:K-1}$ denotes the vector formed by the first $K-1$ components of the one-hot encoding.

We have thus shown that the categorical distribution belongs to the multiparameter exponential family.

The main difference from the Bernoulli distribution, when $K>2$, concerns the dimensionality of the parameters: rather than a single natural parameter $\theta$, we have a vector $\boldsymbol\theta$ of dimension $K-1$.

Consequently, the cumulant function $b(\theta)$ is generalized to the multivariate function $A(\boldsymbol\theta)$.

## 4.5 Expected Value and Covariance Matrix

In Chapter 2, we showed that, for the exponential family under the parameterization considered:

$$
\mathbb E[Y]=b'(\theta)
$$

$$
\mathrm{Var}(Y)=a(\phi)b''(\theta)
$$

In the categorical case, the cumulant function depends on a vector of parameters.

Ordinary derivatives are therefore replaced by the **gradient** and the **Hessian matrix**.

For the exponential-family representation just derived, we have:

$$
\boxed{\mathbb E[\mathbf Y_{1:K-1}]=\nabla A(\boldsymbol\theta)}
$$

and:

$$
\boxed{\mathrm{Cov}(\mathbf Y_{1:K-1})=\nabla^2 A(\boldsymbol\theta)}
$$

These relationships are the multivariate generalization of the properties already studied for the Bernoulli distribution. In this exponential-family form, the scale factor is one, so the term $a(\phi)$ from the general GLM variance formula does not appear.

### Deriving the Expected Value

We have:

$$
A(\boldsymbol\theta)=\ln\left(1+\sum_{h=1}^{K-1}e^{\theta_h}\right)
$$

Calculate the partial derivative with respect to $\theta_j$:

$$
\frac{\partial A}{\partial\theta_j}=\frac{\partial}{\partial\theta_j}\ln\left(1+\sum_{h=1}^{K-1}e^{\theta_h}\right)
$$

Applying the chain rule:

$$
\frac{\partial A}{\partial\theta_j}=\frac{\frac{\partial}{\partial\theta_j}\left(1+\sum_{h=1}^{K-1}e^{\theta_h}\right)}{1+\sum_{h=1}^{K-1}e^{\theta_h}}
$$

Since only the term with $h=j$ in the sum depends on $\theta_j$:

$$
\frac{\partial A}{\partial\theta_j}=\frac{e^{\theta_j}}{1+\sum_{h=1}^{K-1}e^{\theta_h}}
$$

We recognize the probability $p_j$ derived earlier:

$$
\boxed{\frac{\partial A}{\partial\theta_j}=p_j}
$$

Therefore:

$$
\boxed{\nabla A(\boldsymbol\theta)=
\begin{pmatrix}
p_1\\
p_2\\
\vdots\\
p_{K-1}
\end{pmatrix}}
$$


This result is consistent with the definition of the indicator variables.

Indeed, for each category:

$$
\begin{aligned}
\mathbb E[Y_j]
&=1\cdot P(Y_j=1)+0\cdot P(Y_j=0)\\
&=p_j
\end{aligned}
$$

Thus, the first derivative of the cumulant function returns the probabilities of the first $K-1$ categories.

The probability of the reference category is determined by the constraint:

$$
p_K=1-\sum_{j=1}^{K-1}p_j
$$

### Deriving the Covariance Matrix

Now consider the second derivatives of $A(\boldsymbol\theta)$.

Since:

$$
\frac{\partial A}{\partial\theta_j}=p_j
$$

we have:

$$
\frac{\partial^2 A}{\partial\theta_h\partial\theta_j}=\frac{\partial p_j}{\partial\theta_h}
$$

Recall that:

$$
p_j=
\frac{e^{\theta_j}}
{1+\sum_{r=1}^{K-1}e^{\theta_r}}
$$

We distinguish two cases.

**Case 1: $j=h$**

Differentiating with respect to $\theta_j$:

$$
\frac{\partial p_j}{\partial\theta_j}=\frac{e^{\theta_j}\left(1+\sum_{r=1}^{K-1}e^{\theta_r}\right)-e^{2\theta_j}
}{
\left(1+\sum_{r=1}^{K-1}e^{\theta_r}\right)^2
}
$$

Recognizing $p_j$, we obtain:

$$
\boxed{
\frac{\partial p_j}{\partial\theta_j}
=p_j(1-p_j)
}
$$

**Case 2: $j\neq h$**

In this case, the numerator $e^{\theta_j}$ does not depend on $\theta_h$.

Applying the quotient rule:

$$
\frac{\partial p_j}{\partial\theta_h}=-\frac{e^{\theta_j}e^{\theta_h}}{\left(1+\sum_{r=1}^{K-1}e^{\theta_r}\right)^2}
$$

Therefore:

$$
\boxed{
\frac{\partial p_j}{\partial\theta_h}
=-p_jp_h
}
$$

The two relationships can be expressed using the Kronecker delta:

$$
\delta_{jh}=
\begin{cases}
1 & \text{if }j=h\\
0 & \text{if }j\neq h
\end{cases}
$$

We obtain:

$$
\boxed{\frac{\partial^2 A}{\partial\theta_j\partial\theta_h}=p_j(\delta_{jh}-p_h)}
$$

This expression gives the entries of the covariance matrix of the indicator vector.

For $j=h$:

$$
\mathrm{Var}(Y_j)=p_j(1-p_j)
$$

For $j\neq h$:

$$
\mathrm{Cov}(Y_j,Y_h)=-p_jp_h
$$

The negative covariances reflect the mutually exclusive nature of the categories: if one category is observed, the others cannot be observed in the same trial.

The matrix can be written in compact form:

$$
\boxed{\mathrm{Cov}(\mathbf Y_{1:K-1})=\mathrm{diag}(\mathbf p_{1:K-1})-\mathbf p_{1:K-1}\mathbf p_{1:K-1}^{\top}}
$$

This result highlights how the multivariate cumulant function encodes both the category probabilities and the structure of their dependencies.

**A note on dimensionality.** The matrix just derived is the covariance matrix of the reduced vector $\mathbf Y_{1:K-1}$ and has dimensions $(K-1)\times(K-1)$. If $p_1,\ldots,p_K>0$, it is positive definite. By contrast, the covariance matrix of the full one-hot vector $\mathbf Y$ is $\mathrm{diag}(\mathbf p)-\mathbf p\mathbf p^\top$; it has dimensions $K\times K$ and is singular because $\sum_jY_j=1$ imposes a linear dependence among the components.

## 4.6 Numerical Example with Three Categories

Consider a categorical distribution with:

$$
K=3
$$

Choose the third category as the reference and define the natural parameters:

$$
\theta_1=\ln 2
$$

$$
\theta_2=\ln 3
$$

By definition:

$$
e^{\theta_1}=\frac{p_1}{p_3}=2
$$

$$
e^{\theta_2}=\frac{p_2}{p_3}=3
$$

We therefore obtain:

$$
p_1=2p_3
$$

$$
p_2=3p_3
$$

Use the normalization constraint:

$$
p_1+p_2+p_3=1
$$

Substituting:

$$
2p_3+3p_3+p_3=1
$$

Hence:

$$
6p_3=1
$$

and:

$$
p_3=\frac16
$$

The other probabilities are:

$$
p_1=2\cdot\frac16=\frac26
$$

$$
p_2=3\cdot\frac16=\frac36
$$

Thus:

$$
\boxed{
\mathbf p=
\left(
\frac26,\frac36,\frac16
\right)
}
$$

Verify that:

$$
\frac26+\frac36+\frac16=1
$$

The cumulant function is:

$$
A(\boldsymbol\theta)=\ln(1+e^{\theta_1}+e^{\theta_2})
$$

Substituting:

$$
A(\boldsymbol\theta)=\ln(1+2+3)
$$

We obtain:

$$
\boxed{
A(\boldsymbol\theta)=\ln 6
}
$$

If we observe the second category:

$$
\mathbf y=(0,1,0)
$$

the exponential-family representation gives:

$$
P(\mathbf Y=\mathbf y)=\exp[y_1\theta_1+y_2\theta_2-A(\boldsymbol\theta)]
$$

Substituting:

$$
P(\mathbf Y=(0,1,0))=\exp[0\cdot\ln2+1\cdot\ln3-\ln6]
$$

Thus:

$$
P(\mathbf Y=(0,1,0))=e^{\ln3-\ln6}
$$

and finally:

$$
\boxed{P(\mathbf Y=(0,1,0))=\frac36=\frac12}
$$

We have verified that the exponential-family representation correctly reproduces the probability of the observed category.

## 4.7 Toward the Multinomial Canonical Link

We can now summarize the results.

The categorical distribution belongs to the multiparameter exponential family and has a vector of $K-1$ natural parameters:

$$
\boxed{
\theta_j=\ln\left(\frac{p_j}{p_K}\right),
\qquad j=1,\ldots,K-1
}
$$

The cumulant function is:

$$
\boxed{
A(\boldsymbol\theta)=
\ln\left(
1+\sum_{j=1}^{K-1}e^{\theta_j}
\right)
}
$$

The category probabilities can be obtained from:

$$
p_j=
\frac{e^{\theta_j}}
{1+\sum_{h=1}^{K-1}e^{\theta_h}}
$$

for $j=1,\ldots,K-1$, while the reference-category probability is:

$$
p_K=
\frac{1}
{1+\sum_{h=1}^{K-1}e^{\theta_h}}
$$

We also showed that:

$$
\mathbb E[\mathbf Y_{1:K-1}]=\nabla A(\boldsymbol\theta)
$$

and:

$$
\mathrm{Cov}(\mathbf Y_{1:K-1})=\nabla^2A(\boldsymbol\theta)
$$

**The key result is that the categorical distribution can be represented using $K-1$ log-ratios of probabilities**, rather than a single logit as in the Bernoulli case.

This structure allows us to generalize the concept of the canonical link from the scalar to the vector case.

In the next chapter, we will define a linear predictor for each of the $K-1$ non-reference categories, setting the predictors equal to the natural parameters.

By inverting the multinomial canonical link, we will obtain the softmax function.

We will also examine why the logit of a reference category can be fixed at zero, the role of the softmax function's invariance under additive shifts, and how binary logistic regression emerges as a special case of the multinomial construction.
