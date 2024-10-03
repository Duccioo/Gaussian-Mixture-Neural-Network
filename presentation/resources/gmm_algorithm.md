# Gaussian Mixture Model (GMM) Algorithm

A Gaussian Mixture Model is a probabilistic model that assumes all the data points are generated from a mixture of a finite number of Gaussian distributions with unknown parameters. Here's a detailed explanation of the GMM algorithm:

## 1. Model Definition

Let $$X = \{x_1, \ldots, x_N\}$$ be a set of N observations. A Gaussian Mixture Model with K components is defined as:

$$p(x) = \sum_{k=1}^K \pi_k \mathcal{N}(x | \mu_k, \Sigma_k)$$

Where:

-> $$ \pi_k $$ is the mixture weight (or probability) of the k-th component

-> $$ \mathcal{N}(x | \mu_k, \Sigma_k)$$ is the probability density function of a Gaussian distribution with mean $$\mu_k$$ and covariance matrix $$\Sigma_k$$

## 2. Parameters

The parameters of a GMM are:

- 1 $$\pi = \{\pi_1, \ldots, \pi_K\}$$: mixture weights
- 2 $$\mu = \{\mu_1, \ldots, \mu_K\}$$: means of the Gaussians
- 3 $$\Sigma = \{\Sigma_1, \ldots, \Sigma_K\}$$: covariance matrices

Subject to the constraint: $$\sum_{k=1}^K \pi_k = 1$$

## 3. Expectation-Maximization (EM) Algorithm

The EM algorithm is used to estimate the parameters of the GMM:

### Initialization

- Initialize the means $$\mu_k$$, covariances $$\Sigma_k$$, and mixing coefficients $$\pi_k$$

### Expectation Step (E-step)

Calculate the responsibilities $$\gamma_{nk}$$ for each data point $$x_n$$ and each Gaussian component k:

$$\gamma_{nk} = \frac{\pi_k \mathcal{N}(x_n | \mu_k, \Sigma_k)}{\sum_{j=1}^K \pi_j \mathcal{N}(x_n | \mu_j, \Sigma_j)}$$

### Maximization Step (M-step)

Update the parameters:

1. Update the means:
   $$\mu_k^{new} = \frac{\sum_{n=1}^N \gamma_{nk} x_n}{\sum_{n=1}^N \gamma_{nk}}$$

2. Update the covariances:
   $$\Sigma_k^{new} = \frac{\sum_{n=1}^N \gamma_{nk} (x_n - \mu_k^{new})(x_n - \mu_k^{new})^T}{\sum_{n=1}^N \gamma_{nk}}$$

3. Update the mixing coefficients:
   $$\pi_k^{new} = \frac{1}{N} \sum_{n=1}^N \gamma_{nk}$$

### Convergence

Repeat the E-step and M-step until the log-likelihood converges or a maximum number of iterations is reached.

The log-likelihood is given by:

$$\ln p(X | \pi, \mu, \Sigma) = \sum_{n=1}^N \ln \left( \sum_{k=1}^K \pi_k \mathcal{N}(x_n | \mu_k, \Sigma_k) \right)$$

## 4. Prediction

For a new data point $$x_{new}$$, the probability that it belongs to the k-th component is:

$$p(k | x_{new}) = \frac{\pi_k \mathcal{N}(x_{new} | \mu_k, \Sigma_k)}{\sum_{j=1}^K \pi_j \mathcal{N}(x_{new} | \mu_j, \Sigma_j)}$$

The overall probability density for $$x_{new}$$ is:

$$p(x_{new}) = \sum_{k=1}^K \pi_k \mathcal{N}(x_{new} | \mu_k, \Sigma_k)$$

This algorithm allows for the estimation of complex probability density functions and can be used for various tasks such as density estimation, clustering, and anomaly detection.
