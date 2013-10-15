import numpy as np
import matplotlib.pyplot as pp

def abs2(vec):
    """ Norm of the vector squared """
    return np.sum(vec*vec)

def k_n_m(xn, xm, thetas):
    return thetas[0]*np.exp(-thetas[1]/2.0 * abs2(xn-xm)) + thetas[2] + thetas[3]*np.dot(xn.T, xm)

def computeK(X, thetas):
    N = X.shape[0]
    K = np.zeros((N,N))    

    for n in xrange(N):
        for m in xrange(N):
            K[n,m] = k_n_m(X[n], X[m], thetas)

    return K

def computeMean(X, t):
    K = computeK(K, thetas?)
    np.dot(K.t


def gp_predictive_distribution(X_train, T_train, X_test, theta, C = None):
    D, N = X_train.shape
    mu = np.zeros(N)
    var = np.zeros(N)
    if not C:
        K = computeK(X_train, theta)
        C = K + 0.01 * np.identity(K.shape) #sigma?

    for n in xrange(X_test.shape[1]):
        c = k_n_m(X_test[i], X_test[i]) + 0.01
        k = np.empty(N)
        for i in xrange(N):
            k[n] = k_n_m(X_test[n], X_train[i])
        mu[n] = np.dot(np.dot(k.T, np.inverse(C)), T_train)
        var[n] = c - np.dot(np.dot(k.T, np.inverse(C)), k)

return mean, variancesq
