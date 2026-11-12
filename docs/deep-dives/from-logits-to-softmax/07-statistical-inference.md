# 7. Statistical Inference and Predictive Uncertainty

In the previous chapters, we showed how the categorical distribution can be parameterized using the softmax function and how its likelihood naturally leads to categorical cross-entropy.

Consider a model that takes a feature vector $x$ and produces $K$ logits through a function parameterized by $\psi$:

$$
\mathbf z=f_\psi(x)
$$

Applying the softmax, we obtain:

$$
\boxed{
P(Y=j\mid x,\psi)=
\frac{e^{z_j}}{\sum_{h=1}^{K}e^{z_h}}
}
$$

This expression defines a conditional distribution of the response variable over the $K$ categories.

However, the fact that a model returns a probability distribution does not automatically mean that these probabilities are reliable or that they capture all the uncertainty associated with a prediction.

To understand this distinction, we need to separate at least three questions:

1. How are the model parameters estimated?
2. How uncertain are the resulting estimates?
3. How reliable are the probabilities produced for new observations?

These questions lead to the concepts of frequentist inference, Bayesian inference, and probabilistic calibration.

## 7.1 Probabilistic Models and Parameter Uncertainty

The first distinction concerns two different levels of probabilistic modeling.

**Distribution of the response variable**

In a classification model, we assume that, conditional on the observation's features and the model parameters, the response follows a categorical distribution:

$$
Y\mid x,\psi
\sim
\mathrm{Categorical}(\mathbf p_\psi(x))
$$

where:

$$
\mathbf p_\psi(x)=
\mathrm{softmax}(f_\psi(x))
$$

This distribution describes the probabilities of the possible outcomes when the model parameters are treated as fixed.

**Parameter uncertainty**

A different question is how much we know about the parameters $\psi$ from the available data.

We may have estimated the parameters from a limited sample, or there may be several parameter configurations compatible with the observed data.

The categorical output distribution does not, by itself, represent this second form of uncertainty.

We must therefore distinguish:

$$
P(Y\mid x,\psi)
$$

from a description of the uncertainty associated with $\psi$.

Statistical inference provides tools for addressing the latter problem, although the interpretation differs according to the approach adopted.

## 7.2 Frequentist Inference and Maximum Likelihood

In the frequentist framework, model parameters are treated as fixed but unknown quantities.

Consider, for example, binary logistic regression:

$$
Y_i\mid x_i,\beta
\sim
\mathrm{Bernoulli}(p_i)
$$

where:

$$
p_i=\sigma(x_i^\top\beta)
$$

The vector $\beta$ contains the model coefficients.

Assuming conditionally independent observations, we can construct the likelihood function:

$$
\boxed{
L(\beta;\mathcal D)=
\prod_{i=1}^{n}
f(y_i\mid x_i,\beta)
}
$$

where:

$$
\mathcal D=\{(x_i,y_i)\}_{i=1}^{n}
$$

denotes the observed sample.

The maximum likelihood estimator is defined as:

$$
\boxed{
\hat\beta_{\mathrm{MLE}}=
\arg\max_\beta L(\beta;\mathcal D)
}
$$

Equivalently, we can maximize the log-likelihood:

$$
\hat\beta_{\mathrm{MLE}}=
\arg\max_\beta\ell(\beta)
$$

For the categorical model with softmax, we have already shown that minimizing cross-entropy is equivalent to maximizing the log-likelihood when no additional regularization terms are present.

### The Sampling Distribution of an Estimator

In the frequentist approach, the parameter $\beta$ is considered fixed.

The estimator $\hat\beta$, however, is a random variable because it depends on the sample used to estimate it.

If we repeated the sampling and estimation procedures under the same conditions, we would generally obtain different values of $\hat\beta$.

The distribution of these values is called the **sampling distribution of the estimator**.

Under suitable regularity conditions, correct model specification, parameter identifiability, and with a sufficiently large sample, the maximum likelihood estimator is approximately normally distributed:

$$
\boxed{
\hat\beta_{\mathrm{MLE}}
\overset{\cdot}{\sim}
\mathcal N\left(
\beta_0,
I_n(\beta_0)^{-1}
\right)
}
$$

where $\beta_0$ denotes the true parameter value and $I_n(\beta_0)$ is the Fisher information matrix for the entire sample, here considered conditional on the observed covariates.

The symbol $\overset{\cdot}{\sim}$ denotes a distributional approximation valid for sufficiently large samples, **not an exact equality of distributions**. In particular, the information matrix must be nonsingular, and the estimator must satisfy the usual existence and regularity conditions.

### The Fisher Information Matrix

Under appropriate regularity conditions, the Fisher information matrix can be defined as:

$$
\boxed{
I_n(\beta)=-\mathbb E_\beta
\left[
\frac{\partial^2\ell(\beta)}
{\partial\beta\,\partial\beta^\top}
\right]
}
$$

This matrix describes the amount of information contained in the likelihood about the parameters.

For example, in binary logistic regression with conditionally independent observations, the matrix takes the form:

$$
\boxed{
I_n(\beta)
=
\sum_{i=1}^{n}
p_i(1-p_i)x_ix_i^\top
}
$$

This expression follows from the curvature of the log-likelihood and the relationship already derived for the Bernoulli distribution:

$$
\mathrm{Var}(Y_i\mid x_i)=p_i(1-p_i)
$$

When the information matrix is invertible, its inverse provides, **under the model assumptions and within the asymptotic approximation**, an estimate of the covariance matrix of the maximum likelihood estimator.

For canonical logistic regression with covariates treated as fixed, the negative Hessian of the log-likelihood coincides with the expression given for $I_n(\beta)$: in this case, the observed and expected information have the same functional form.

In practice, the matrix is evaluated at the estimated parameter values. If the model is misspecified, the inverse Fisher information alone may not provide a valid covariance estimate; under appropriate conditions, robust *sandwich* estimators can be considered, for example.

### Standard Errors and Confidence Intervals

The asymptotic approximation allows us to quantify estimation uncertainty using standard errors.

For a generic coefficient $\beta_j$:

$$
SE(\hat\beta_j)
\approx
\sqrt{
\left[I_n(\hat\beta)^{-1}\right]_{jj}
}
$$

An approximate $95\%$ Wald confidence interval can be constructed as:

$$
\boxed{
\hat\beta_j
\pm
1.96\,SE(\hat\beta_j)
}
$$

The frequentist interpretation concerns the procedure: if we repeatedly sampled and constructed the interval under the same conditions, approximately $95\%$ of those intervals would contain the true parameter, provided that the approximation is adequate.

In logistic regression, we can exponentiate the interval endpoints to obtain an approximate confidence interval for the odds ratio:

$$
OR_j=e^{\beta_j}
$$



$$
\boxed{
IC_{95\%}(OR_j)
\approx
\left[
 e^{\hat\beta_j-1.96\,SE(\hat\beta_j)},\;
 e^{\hat\beta_j+1.96\,SE(\hat\beta_j)}
\right]
}
$$

The exponential function is strictly increasing: transforming both endpoints preserves the coverage interpretation, within the accuracy of the approximation used.

In the multinomial case, with reference category $K$, the same procedure can be applied to coefficients associated with the log-ratios:

$$
\ln\left(\frac{p_j}{p_K}\right)=x^\top\beta_j
$$

To obtain valid inference, however, we must use an identifiable parameterization, for example by fixing the coefficients of the reference category.

### Limitations and Conditions of Applicability

Inference procedures based on maximum likelihood require suitable assumptions.

Potential difficulties include complete separation in logistic regression, small sample sizes, non-identifiability of parameters, and model misspecification.

For deep neural networks, high dimensionality, parameterization symmetries, and the nonconvex structure of the optimization problem can make direct application of classical inference techniques considerably more difficult.

The existence of a likelihood function therefore does not, by itself, guarantee that reliable confidence intervals can easily be constructed for every parameter.

## 7.3 Bayesian Inference: A Distribution over Parameters

The Bayesian approach offers a different representation of uncertainty.

Rather than describing only the sampling distribution of an estimator, we introduce a probability distribution directly over the model parameters.

This distribution is updated in light of the observed data.

### The Prior Distribution

Consider a logistic model with coefficient vector $\beta$.

For example, we can introduce a multivariate normal distribution:

$$
\boxed{
\beta\sim\mathcal N(\mathbf 0,\tau^2I)
}
$$

This distribution represents the *prior*, that is, the prior distribution over the parameters.

The parameter $\tau>0$ controls its dispersion (while $I$ is the identity matrix of the appropriate dimension).

In multinomial regression, the prior can be specified over the $K-1$ free coefficient vectors using an identifiable parameterization.

### The Posterior Distribution

After observing the sample $\mathcal D$, we apply Bayes' theorem:

$$
\boxed{
p(\beta\mid\mathcal D)=
\frac{
p(\mathcal D\mid\beta)p(\beta)
}{
p(\mathcal D)
}
}
$$

The terms represent:

| Term | Meaning |
|---|---|
| $p(\beta)$ | Prior distribution over the parameters |
| $p(\mathcal D\mid\beta)$ | Data likelihood |
| $p(\mathcal D)$ | Evidence or marginal likelihood |
| $p(\beta\mid\mathcal D)$ | Posterior distribution |

The posterior distribution combines the information encoded by the prior with that contained in the observed data. The normalizing constant is the evidence:

$$
p(\mathcal D)=
\int p(\mathcal D\mid\beta)p(\beta)\,d\beta
$$

Here we assume that the prior is proper and that the integral is finite and positive.

Unlike a point estimate, this representation allows us to describe a range of plausible parameter values and their associated probabilistic uncertainty.

From the posterior, we can derive point estimates, credible intervals, and predictive distributions.

### The Bayesian Predictive Distribution

Consider a new observation with covariates $x_*$.

If the parameters were known exactly, the probability of a given category would be:

$$
P(Y_*=j\mid x_*,\beta)
$$

In the Bayesian approach, instead, we account for parameter uncertainty by integrating over the posterior distribution:

$$
\boxed{
P(Y_*=j\mid x_*,\mathcal D)=
\int
P(Y_*=j\mid x_*,\beta)
p(\beta\mid\mathcal D)\,d\beta
}
$$

For a neural classifier parameterized through softmax, we instead use $\psi$ to denote the complete set of network weights:

$$
\boxed{
P(Y_*=j\mid x_*,\mathcal D)=
\int
\mathrm{softmax}(f_\psi(x_*))_j
p(\psi\mid\mathcal D)\,d\psi
}
$$

The prediction therefore does not depend on a single estimated parameter configuration. Instead, it integrates predictions produced by different configurations, weighted according to their posterior probabilities.

Importantly, in general:

$$
\mathbb E_{\psi\mid\mathcal D}\!\left[
\mathrm{softmax}(f_\psi(x))
\right]
\neq
\mathrm{softmax}\!\left(
\mathbb E_{\psi\mid\mathcal D}[f_\psi(x)]
\right)
$$

The softmax is a nonlinear function.

Therefore, in general, we cannot obtain the Bayesian predictive distribution simply by applying the softmax to the mean of the logits.

For logistic models and neural networks, the integrals required to compute posterior and predictive distributions are often unavailable in closed form.

Numerical methods are therefore used, including Monte Carlo sampling, MCMC techniques, and variational approximations.

## 7.4 Aleatoric and Epistemic Uncertainty

The distinction between the conditional distribution of the response and parameter uncertainty allows us to introduce two fundamental concepts in machine learning.

### Aleatoric Uncertainty

**Aleatoric uncertainty** refers to outcome variability that remains even when the model is fixed and its parameters are known.

For example, suppose a classifier assigns the following probabilities to two categories:

$$
\mathbf p=(0.5,0.5)
$$

The prediction is intrinsically uncertain: according to the model, both categories are equally likely.

Aleatoric uncertainty may arise from the stochastic nature of the phenomenon, overlap between classes, or information unavailable in the covariates being considered.

It cannot necessarily be eliminated by increasing the number of observations used to estimate the parameters.

### Epistemic Uncertainty

**Epistemic uncertainty**, by contrast, concerns incomplete knowledge of the model or its parameters.

It may be particularly relevant when little data is available, some regions of the feature space are poorly represented, or the model is applied to situations different from those encountered during training.

Access to more relevant data can help reduce this form of uncertainty.

### An Example Illustrating the Difference

Consider two possible situations in a binary classification problem.

In the first case, several plausible model configurations all produce similar probabilities:

$$
\mathbf p_\beta\approx(0.5,0.5)
$$

The predictions are uncertain, but the different models largely agree.

In the second case, two equally plausible configurations instead produce:

$$
\mathbf p_{\beta_1}=(0.9,0.1)
$$



$$
\mathbf p_{\beta_2}=(0.1,0.9)
$$

If we assign equal weight to the two configurations, the mean predictive distribution is:

$$
\overline{\mathbf p}=
\frac12\mathbf p_{\beta_1}+
\frac12\mathbf p_{\beta_2}
$$

Therefore:

$$
\boxed{
\overline{\mathbf p}=(0.5,0.5)
}
$$

The mean predictive distribution is identical to that in the first case.

However, the source of uncertainty is different: in the second case, the models strongly disagree.

This example illustrates why **the mean predictive distribution alone does not always allow us to distinguish outcome uncertainty from disagreement caused by parameter uncertainty**.

Making this distinction requires additional information about the distribution of predictions or the model parameters.

### A Quantitative Decomposition in the Binary Case

We can express this difference more precisely using the **law of total variance**, when a posterior distribution over parameters is available. Let $Z=\mathbf{1}_{\{Y=1\}}$ denote the indicator of a category, and let $p_\beta=P(Z=1\mid x,\beta)$ be the probability predicted by a parameter configuration $\beta$.

Conditional on the data $\mathcal D$ and the covariates $x$, we obtain:

$$
\boxed{
\mathrm{Var}(Z\mid x,\mathcal D)=
\underbrace{\mathbb E_{\beta\mid\mathcal D}
\left[p_\beta(1-p_\beta)\right]}_{\text{mean conditional variability}}+
\underbrace{\mathrm{Var}_{\beta\mid\mathcal D}(p_\beta)}_{\text{between-prediction variability}}
}
$$

In the first example, with $p_\beta=0.5$ for every plausible model, the two terms are $0.25$ and $0$, respectively. In the second example, with $p_{\beta_1}=0.9$ and $p_{\beta_2}=0.1$ and equal weights, they are $0.09$ and $0.16$. In both cases, the total predictive variance is $0.25$, but its decomposition is different.

This provides one possible **aleatoric/epistemic decomposition within the specified Bayesian framework**. It is not a universal measure of epistemic uncertainty, because it depends on the model class and the distribution over parameters adopted.

## 7.5 Probability Calibration

Another important issue concerns the reliability of the probabilities produced by the model.

Consider a neural classifier that returns the following probabilities for an observation:

$$
\mathbf p=(0.9,0.07,0.03)
$$

The first category is assigned a probability of $90\%$.

However, from the application of softmax alone, we cannot conclude that the predicted category is actually correct in approximately $90\%$ of observations to which the model assigns similar confidence.

Softmax guarantees that probabilities are positive and sum to one, but it does not guarantee that they are **calibrated against observed frequencies**.

### What Does Calibration Mean?

Intuitively, a model is well calibrated if, among predictions assigned a given probability, the corresponding events occur with a frequency consistent with that probability.

For example, in a binary classification problem, consider a set of observations to which the model assigns a probability of success close to $0.8$.

If the model is well calibrated, we would expect approximately 80% of these cases to have a positive outcome, across a sufficiently large number of comparable observations.

Calibration therefore concerns the relationship between the probabilities reported by a model and the empirical frequencies of the events.

In multiclass problems, it is useful to distinguish at least two assessments: **classwise calibration**, referring to probabilities assigned to a particular category, and **confidence calibration**, referring to the model's maximum predicted probability and the frequency with which the predicted category is correct. These criteria are not equivalent: a model can be well calibrated according to one but not the other.

### Evaluation Tools

Calibration can be investigated using several tools.

**Reliability diagram**

A reliability diagram compares, across groups of predictions, the mean probabilities assigned by the model with the observed frequencies.

A perfectly calibrated model tends to produce points close to the plot's diagonal, allowing for sampling variability.

**Brier score**

The Brier score measures the squared error between the predicted probabilities and the indicators of the observed categories.

For $n$ observations and $K$ categories, one multiclass formulation is:

$$
\boxed{
BS=
\frac1n
\sum_{i=1}^{n}
\sum_{j=1}^{K}
(p_{ij}-y_{ij})^2
}
$$

Lower values indicate better performance according to this metric.

The formula shown is **one convention for the multiclass Brier score**: some sources also divide the result by $K$ (or rescale it in other ways). These factors change the scale, not the ranking of models when applied consistently. In the binary case, summing over both categories yields twice the squared error commonly used for the positive-class probability alone.

The Brier score jointly evaluates different aspects of probabilistic prediction quality and is not exclusively a calibration metric.

**Log-loss**

In the categorical case, log-loss coincides with the cross-entropy studied earlier:

$$
\boxed{
\mathcal L=
-\frac1n
\sum_{i=1}^{n}
\sum_{j=1}^{K}
y_{ij}\ln p_{ij}
}
$$

This metric particularly strongly penalizes predictions that assign very low probabilities to events that actually occur.

Here too, a good log-loss does not automatically imply perfect calibration.

### Temperature Scaling

One technique used to improve the calibration of some neural classifiers is *temperature scaling*.

Instead of applying softmax directly to the logits, we introduce a positive parameter $T$:

$$
\boxed{
p_j(T)=
\frac{e^{z_j/T}}
{\sum_{h=1}^{K}e^{z_h/T}}
}
$$

The parameter $T$ controls how concentrated the distribution is:

- If $T=1$, we recover the original softmax.
- If $T>1$, the probabilities tend to become more uniformly distributed.
- If $0<T<1$, the probabilities tend to concentrate more strongly on categories with higher logits.

The value of $T$ can be estimated by minimizing, for example, log-loss on a separate **calibration set**, distinct from the data used to train the model weights. Final evaluation must be performed on data used for neither training nor temperature selection.

Because the transformation divides all logits by the same positive constant, it does not change the ordering of the categories and therefore does not change the class with the highest predicted probability.

It can, however, change the quality of the predicted probabilities. In particular, as $T\to+\infty$, the distribution approaches the uniform distribution over the $K$ categories; as $T\to0^+$, it concentrates on the categories with the maximum logit.

Temperature scaling is not a universal solution to calibration problems, especially when observations differ substantially from those used to train or calibrate the model.

## 7.6 Implications for Neural Networks and Transformers

In conventional deep learning, network parameters are generally estimated by minimizing a loss function:

$$
\boxed{
\hat\psi=
\arg\min_\psi\mathcal L(\psi)
}
$$

The result is an optimized parameter configuration.

Using this configuration, we can produce logits and probabilities:

$$
\mathbf p_{\hat\psi}(x)=
\mathrm{softmax}(f_{\hat\psi}(x))
$$

This procedure defines a conditional categorical distribution, but does not automatically produce a probability distribution over the network parameters.

To investigate additional forms of uncertainty, methods such as model ensembles, bootstrap techniques, Bayesian approximations, or dropout-based techniques can be used, depending on the problem.

These procedures rely on different assumptions and have different limitations; none automatically guarantees an accurate quantification of uncertainty.

### The Case of Language Models

In decoder-only Transformers, the model produces a probability distribution over the next token:

$$
P(w_{t+1}=j\mid w_{\leq t},\hat\psi)
$$

This distribution allows us to select or sample the next token.

It should not, however, automatically be interpreted as a measure of the overall reliability of the generated text.

In particular, assigning a high probability to a token does not guarantee the factual correctness of the statement in which that token appears.

Token probabilities describe the model's predictive behavior relative to the learned language distribution, whereas truthfulness, reliability, and the soundness of the generated information require additional evaluation criteria.

We must therefore distinguish predictive probabilities, calibration, parameter uncertainty, and the semantic reliability of the output.

## 7.7 Summary

The probabilistic representation of the response provides tools that go beyond simply assigning a category.

Through the likelihood, we can estimate parameters, construct inference procedures, and quantitatively evaluate prediction quality.

We have also seen that uncertainty can concern different aspects of the model.

| Concept | Main question |
|---|---|
| Predictive probability | What probability is assigned to each outcome? |
| Frequentist inference | How much does the estimator vary across samples, under the model assumptions? |
| Bayesian inference | Which parameter values are plausible given the data? |
| Aleatoric uncertainty | What outcome variability remains, given the model? |
| Epistemic uncertainty | How incomplete is our knowledge of the model? |
| Calibration | Are predicted probabilities consistent with observed frequencies? |

The key distinction to retain is that **softmax produces a valid probability distribution, but does not by itself guarantee a complete and reliable quantification of uncertainty**.

Prediction quality also depends on model specification, the availability and representativeness of data, the estimation process, and the adequacy of evaluation procedures.

This completes the connection between the statistical derivation of softmax, parameter estimation, and the interpretation of the probabilities produced by neural networks.

In the concluding chapter, we will summarize the entire theoretical development, from the exponential family to the softmax function, highlighting the connections we will encounter again in the implementation of the decoder-only Transformer.
