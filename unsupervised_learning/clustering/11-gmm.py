#!/usr/bin/env python3

"""calculates the gmm using sklearn"""
import sklearn.mixture as sk


def gmm(X, k):
    """calculates the gmm using sklearn"""
    gmm = sk.GaussianMixture(n_components=k)
    gmm.fit(X)
    pi = gmm.weights_
    m = gmm.means_
    S = gmm.covariances_
    clss = gmm.predict(X)
    bic = gmm.bic(X)
    return pi ,m , S, clss, bic
