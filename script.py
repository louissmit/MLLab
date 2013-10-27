# -*- coding: utf-8 -*-
"""
Created on Wed Oct 09 12:08:57 2013

@author: AlbertoEAF
"""
from methods import *
import matplotlib.pyplot as pp

X_train, T_train, X_true, Y_true, X_test = data_gen(15,30,sigma=None)
THETAS = np.array([[1,4,0,0], [9,4,0,0], [1,64,0,0], [1,0.25,0,0], [1,4,10,0], [1,4,0,5]])

#_________________________________________________________________________
# Sampling from the Gaussian process prior
# We show for all 6 ThetaCompositions one sample from a Gaussian process prior (GPP). Also shown is the uncertainty of the 
# GPP. In contrast to the book we only show one GPP sample not 5.

show_sample_kernels(X_test, THETAS)
# gp_plot(X_train, T_train,X_true, Y_true, X_test, 100, THETAS)
pp.show()

#_________________________________________________________________________
