# -*- coding: utf-8 -*-
"""
Created on Wed Oct 09 12:08:57 2013

@author: AlbertoEAF
"""
from methods import *
import matplotlib.pyplot as pp

X_train, T_train, X_true, Y_true, X_test = data_gen(15,30,sigma=None)
THETAS = np.array([[1,4,0,0], [9,4,0,0], [1,64,0,0], [1,0.25,0,0], [1,4,10,0], [1,4,0,5]])
sigma = 0.1

#_________________________________________________________________________
# Sampling from the Gaussian process prior
# We show for all 6 ThetaCompositions one sample from a Gaussian process prior (GPP). Also shown is the uncertainty of the 
# GPP. In contrast to the book we only show one GPP sample not 5.

# show_sample_kernels(X_test, THETAS)
# gp_plot(X_train, T_train,X_true, Y_true, X_test, 100, THETAS)
# pp.show()

#_________________________________________________________________________
# 3.2 - Performs the grid-search

s = np.linspace(0,5,5)
theta_search_space = [s, s, s, s]
best, worst = grid_search(X_train, T_train, sigma, theta_search_space)
gp_plot(X_train, T_train,X_true, Y_true, X_test, 100, [best[1], worst[1]])
# plot_log_likelihood(X_train, T_train, [best[1], worst[1]])
pp.show()




#_________________________________________________________________________






