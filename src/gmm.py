"""
This file contains code for clustering all examples in X into K clusters,
using GMM model, and Gibbs Sampling algorithm.
"""

import numpy as np
from bayes_gmm.fbgmm import FBGMM
from bayes_gmm.niw import NIW

def gmm(X,
        K=4,
        n_iter=100,
        alpha=1.0,
        mu_scale=4.0,
        var_scale=0.5,
        covar_scale=0.7):
    N, D = X.shape

    # Initialize prior
    m_0 = np.zeros(D)
    k_0 = covar_scale**2/mu_scale**2
    v_0 = D + 3
    S_0 = covar_scale**2*v_0*np.ones(D)
    prior = NIW(m_0, k_0, v_0, S_0)

    # Setup FBGMM
    fbgmm = FBGMM(X, prior, alpha, K, "rand")

    # Perform Gibbs sampling
    record = fbgmm.gibbs_sample(n_iter)

    K = fbgmm.components.K

    # Augmenting X by adding cluster means as new points
    mus = np.zeros(shape=(K, D))
    for k in range(fbgmm.components.K):
        mu, _ = fbgmm.components.rand_k(k)
        mus[k,:] = mu

    return fbgmm.components.assignments, mus
