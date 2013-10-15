# -*- coding: utf-8 -*-
"""
Created on Wed Oct 09 12:08:57 2013

@author: AlbertoEAF
"""
from methods import *

#_________________________________________________________________________
# Sampling from the Gaussian process prior
show_sample_kernels()

#_________________________________________________________________________
# Predictive distribution

theta = np.array([[1,4,0,0], [9,4,0,0], [1,64,0,0], [1,0.25,0,0], [1,4,10,0], [1,4,0,5]])

# two data points
X_train, T_train, X_test, y_train= data_gen(2,100,sigma=None)
mu, var =gp_predictive_distribution(X_train, T_train, X_test, theta)
gp_plot( X, y_train, mu, var, x_train, t_train, theta, beta )###needs to be edited tomorrow
# ten data points





