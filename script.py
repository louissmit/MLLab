# -*- coding: utf-8 -*-
"""
Created on Wed Oct 09 12:08:57 2013

@author: AlbertoEAF
"""
from methods import *

theta = np.array([[1,4,0,0], [9,4,0,0], [1,64,0,0], [1,0.25,0,0], [1,4,10,0], [1,4,0,5]])


N_train = 2
N_test = 100
sigma=0.1
X_train, T_train, X_test, y_train = data_gen(N_train,N_test,sigma)

#_________________________________________________________________________
# Predictive distribution


mu, var = gp_predictive_distribution(X_train, T_train, X_test, theta[0], sigma)
#gp_plot( X, y_train, mu, var, x_train, t_train, theta, beta )###needs to be edited tomorrow
# ten data points


#_________________________________________________________________________
# Sampling from the Gaussian process prior
show_sample_kernels(N_test, X_test, mu, 0)

#_________________________________________________________________________
# Learning the hyperparameters
    

  
K = computeK(X_train, theta[0])
C = computeC(K, sigma) 
Cinv=np.linalg.inv(C)

#....

#_________________________________________________________________________
# 3.2 - Performs the grid-search

def print_log_likelihood_result (result): 
    print "Log-Likelihood:",result[0], "   Thetas:", result[1]

 def grid_search(X_train, T_train):   
	#Choose the right values!! This search space might not be that relevant
	thetas = [np.linspace(0,2,6), # theta0
	          np.linspace(0,2,6), # theta1
	          np.linspace(0,2,6), # theta2
	          np.linspace(0,2,6)] # theta3

	theta_combinations = gen_theta_combinations(thetas)
	sigma=0.1
	likelihood_results = []
	for theta_combination in theta_combinations:
	    likelihood_result, C, Cinv = gp_log_likelihood( X_train, T_train, theta_combination, sigma, C = None, invC = None )
	    likelihood_results.append( (likelihood_result,theta_combination) )
	    
	likelihood_results = sorted(likelihood_results, key=lambda likelihood_result: likelihood_result[0])

	# This is usually very big but it is asked.
	for r in likelihood_results:
	    print_log_likelihood_result(r)

	print "Grid-search Best result:"
	print_log_likelihood_result(likelihood_results[-1])
	print "Grid-search Worst result:"
	print_log_likelihood_result(likelihood_results[0])

	print "plot-best-and-worst-results-KAREN :P:P(...)"

#_________________________________________________________________________






