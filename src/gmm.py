"""
This file contains code for clustering all examples in X into K clusters,
using GMM model, and Gibbs Sampling algorithm.
"""

import numpy as np
from bayes_gmm.fbgmm import FBGMM
from bayes_gmm.niw import NIW

from numpy.random import multinomial
from scipy.stats import dirichlet
from scipy.stats import multivariate_normal

def gmm(X,
        K=4,
        n_iter=100,
        alpha=1.0,
        mu_scale=4.0,
        var_scale=0.5,
        covar_scale=0.7,
        posterior_predictive_check=False):
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

    mus = np.zeros(shape=(K, D))
    covars = [ np.zeros((D,D)) for i in range(0,K) ]
    for k in range(fbgmm.components.K):
        mu, var = fbgmm.components.rand_k(k)
        mus[k,:] = mu
        covars[k] = np.diag(var)

    # Generate new points for posterior predictive check
    # Generate the same number of points as N
    if posterior_predictive_check:
        np.random.seed(1)
        rstate = 1
        alphas = (alpha / K) + fbgmm.components.counts
        pis = dirichlet.rvs(alphas, random_state=rstate)[0]
        Z = np.zeros(N, dtype=np.uint32)
        X = np.zeros((N, D))
        for n in range(N):
            Z[n] = np.floor(np.argmax(multinomial(1, pis)))
            X[n] = multivariate_normal.rvs(mean=mus[Z[n]], cov=covars[Z[n]])

        return fbgmm.components.assignments, mus, (X, Z)

    return fbgmm.components.assignments, mus
