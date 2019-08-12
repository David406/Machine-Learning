"""Mixture model using EM"""
from typing import Tuple
import numpy as np
from common import GaussianMixture



def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment
    """
    n, d = X.shape
    K, _ = mixture.mu.shape
    post = np.zeros((n, K))
    LL = 0.0
    
    for i in range(n):
        tiled_vector = np.tile(X[i,:], (K,1))
        # Compute p(x^(i)|k) = N(x^(i); mu_k, sigma2_k) for all k
        post[i,:] = np.exp(-((tiled_vector - mixture.mu)**2).sum(axis=1) / (2*mixture.var)) / np.power(2 * np.pi * mixture.var, d/2)
        # Compute p(x^(i)|k)*p(k) for all k
        post[i,:] = post[i,:] * mixture.p
        pxi = np.sum(post[i,:])
        LL += np.log(pxi)
        post[i,:] /= pxi
        
    return post, LL
        
    
    

def mstep(X: np.ndarray, post: np.ndarray) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    n, K = post.shape
    _, d = X.shape
    n_hat = np.sum(post, axis=0)
    p = n_hat/n
    
    mu = np.zeros((K,d))
    var = np.zeros(K)
    
    for k in range(K):
        mu[k,:] = post[:,k] @ X / post[:,k].sum()
        var[k] = ((X - mu[k,:])**2).sum(axis=1) @ post[:,k] / (d * post[:,k].sum())
        
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
        #print(LL)
        mixture = mstep(X, post)
        
    return mixture, post, LL
