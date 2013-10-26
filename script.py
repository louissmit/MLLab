# -*- coding: utf-8 -*-
"""
Created on Wed Oct 09 12:08:57 2013

@author: AlbertoEAF
"""
from methods import *

THETAS = np.array([[1,4,0,0], [9,4,0,0], [1,64,0,0], [1,0.25,0,0], [1,4,10,0], [1,4,0,5]])


N_train = 2
N_test = 100
sigma=0.1
X_train, T_train, x_true, y_true, X_test = data_gen(N_train, N_test, sigma)



#_________________________________________________________________________
# Predictive distribution

#mu, var = gp_predictive_distribution(X_train, T_train, x_true, THETAS[0], sigma)
#gp_plot( X, y_train, mu, var, x_train, t_train, THETAS, beta )###needs to be edited tomorrow
# ten data points


#_________________________________________________________________________
# Sampling from the Gaussian process prior
#show_sample_kernels(N_test, X_test, mu, 0)

#_________________________________________________________________________
# Learning the hyperparameters
  
#K = computeK(X_train, THETAS[0])
#C = computeC(K, sigma) 
#Cinv=np.linalg.inv(C)

#....

#_________________________________________________________________________
# 3.2 - Performs the grid-search


theta_search_space = [np.linspace(0,2,6), np.linspace(0,2,6), np.linspace(0,2,6), np.linspace(0,2,6)]
grid_search(X_train, T_train, sigma, theta_search_space)


#_________________________________________________________________________






