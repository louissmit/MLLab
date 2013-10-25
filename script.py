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


mu, var = gp_predictive_distribution(X_train, T_train, X_test, theta[0])
gp_plot( X, y_train, mu, var, x_train, t_train, theta, beta )###needs to be edited tomorrow
# ten data points


#_________________________________________________________________________
# Sampling from the Gaussian process prior
show_sample_kernels(N_test,mu)

#_________________________________________________________________________
# Learning the hyperparameters
    

  
K = computeK(X_train, theta[0])
C = K + 0.01 * np.identity(N_train) 
Cinv=np.inverse(C)

theta_combinations = []

theta0_values = [0,1,3]
theta1_values = [0,1,3]
theta2_values = [0,1,3]
theta3_values = [0,1,3]

for t0 in theta0_values:
    for t1 in theta1_values:
        for t2 in theta2_values:
            for t3 in theta3_values:
                    theta_combinations.append( (t0,t1,t2,t3) )

likelihood_results = {}
for combination in theta_combinations:
    likelihood_results[combination] = gp_log_likelihood( X_train, T_train, theta, C = None, invC = None )

print "sort-dictionary-thingy(...)"
print "plot-results-KAREN :P:P(...)"





