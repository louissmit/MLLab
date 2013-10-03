# -*- coding: utf-8 -*-
"""
Created on Thu Oct 03 20:59:43 2013

@author: AlbertoEAF
"""

from methods import *
import matplotlib.pyplot as plt

# load data
(x_train, t_train), (x_valid, t_valid), (x_test, t_test) = load_mnist()
# plot_digits(x_train[0:8], numcols=4)

N, M = x_train.shape
K, L = x_valid.shape
J = 10
b = np.zeros(J)
W = np.zeros((M,J))

def logreg_gradient(x, t, W, b):
    M,J=W.shape

    Z = np.sum(np.exp(np.dot(W.T, x) + b))

    grad_b = -np.exp(np.dot(W.T, x) + b) / Z
    grad_b[t]+=1

    grad_W = np.outer(x, grad_b.T)

    return grad_b, grad_W

def logreg_gradient2(x, t, W, b):
    M,J=W.shape

    WT = W.T

    Z = np.sum(np.exp(np.dot(WT, x) + b))

    grad_b = -np.exp(np.dot(WT, x) + b) / Z
    grad_b[t]+=1

    grad_W = np.outer(x, grad_b.T)

    return grad_b, grad_W

import time



I = 10    

m = N

t0a = time.clock()
for n in xrange(m):
    logreg_gradient(x_train[n],t_train[n],W,b)
print "old time: ", time.clock() - t0a

t0b = time.clock()
for n in xrange(m):
    logreg_gradient2(x_train[n],t_train[n],W,b)
print "new time: ", time.clock() - t0b