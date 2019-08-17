"""Mixture model for matrix completion"""
from typing import Tuple
import numpy as np
from scipy.special import logsumexp
from common import GaussianMixture


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment

    """
    n, d = X.shape
    K, _ = mixture.mu.shape
    post = np.zeros((n,K))
    
    for i in range(n):
        # Decide positions with valid ratings
        valid = (X[i,:] > 0)
        tiled_vector = np.tile(X[i,:], (K,1))
        
        # Compute f(u,j) for all j
        post[i,:] = np.log(mixture.p) \
                    - ((tiled_vector - mixture.mu)**2) @ valid / (2*mixture.var) \
                    - (d/2) * np.log(2*np.pi*mixture.var)
    
    # Compute log posterior l(j|u) = f(u,j) - logsumexp(f(u,j))
    LL = logsumexp(post, axis=1, keepdims=True)
    post = post - LL
    # Remove log 
    post = np.exp(post)
    LL = LL.sum()
    
    return post, LL



def mstep(X: np.ndarray, post: np.ndarray, mixture: GaussianMixture,
          min_variance: float = .25) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        post: (n, K) array holding the soft counts
            for all components for all examples
        mixture: the current gaussian mixture
        min_variance: the minimum variance for each gaussian

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    n, d = X.shape
    _, K = post.shape
    
    p = post.sum(axis=0) / n
    
    mu = np.zeros((K, d))
    var = np.zeros(K)
    
    # Valid ratings
    valid = (X > 0)
    Cu = valid.sum(axis=1)
    
    for k in range(K):
        support = (post[:,k].reshape((-1,1)) * valid).sum(axis=0)
        valid_ratings = X * valid
        
        mu[k,:] = post[:,k] @ valid_ratings / support
        mu[k,:] = [x if support[j] >= 1 else mixture.mu[k,j] for j, x in enumerate(mu[k,:])]
        
        var[k] = (((X - mu[k,:]) * valid)**2).sum(axis=1) @ post[:,k] / (Cu @ post[:,k])
        if (var[k] < min_variance):
            var[k] = min_variance
            
    return GaussianMixture(mu, var, p)


def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """
    previous_LL = None
    LL = None
    
    while (previous_LL is None or LL - previous_LL > 1e-6 * np.abs(LL)):
        previous_LL = LL
        post, LL = estep(X, mixture)
        mixture = mstep(X, post, mixture)
        
    return mixture, post, LL


def fill_matrix(X: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
    """Fills an incomplete matrix according to a mixture model

    Args:
        X: (n, d) array of incomplete data (incomplete entries =0)
        mixture: a mixture of gaussians

    Returns
        np.ndarray: a (n, d) array with completed data
    """
    raise NotImplementedError
