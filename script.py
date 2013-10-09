# -*- coding: utf-8 -*-
"""
Created on Wed Oct 09 12:08:57 2013

@author: AlbertoEAF
"""

from methods import *



#sigma = 0.1
#beta  = 1.0 / pow(sigma,2) # this is the beta used in Bishop Eqn. 6.59
N_test = 100
x_test = np.linspace(-1,1,N_test); 
mu_test = np.zeros( N_test )
#
#def true_mean_function( x ):
#    return np.sin( 2*pi*(x+1) )
#
#def add_noise( y, sigma ):
#    return y + sigma*np.random.randn(len(y))
#
#def generate_t( x, sigma ):
#    return add_noise( true_mean_function( x), sigma )
#    
#y_test = true_mean_function( x_test )
#t_test = add_noise( y_test, sigma )
#pp.plot( x_test, y_test, 'b-', lw=2)
#pp.plot( x_test, t_test, 'go')


def gp_plot( x_test, y_test, mu_test, var_test, x_train, t_train, theta, beta ):
    # x_test: 
    # y_test:   the true function at x_test
    # mu_test:   predictive mean at x_test
    # var_test: predictive covariance at x_test 
    # t_train:  the training values
    # theta:    the kernel parameters
    # beta:      the precision (known)
    
    # the reason for the manipulation is to allow plots separating model and data stddevs.
    std_total = np.sqrt(np.diag(var_test))         # includes all uncertainty, model and target noise 
    std_model = np.sqrt( std_total**2 - 1.0/beta ) # remove data noise to get model uncertainty in stddev
    std_combo = std_model + np.sqrt( 1.0/beta )    # add stddev (note: not the same as full)
    
    pp.plot( x_test, y_test, 'b', lw=3)
    pp.plot( x_test, mu_test, 'k--', lw=2 )
    pp.fill_between( x_test, mu_test+2*std_combo,mu_test-2*std_combo, color='k', alpha=0.25 )
    pp.fill_between( x_test, mu_test+2*std_model,mu_test-2*std_model, color='r', alpha=0.25 )
    pp.plot( x_train, t_train, 'ro', ms=10 )
    
THETAS = np.array([[1,4,0,0], [9,4,0,0], [1,64,0,0], [1,0.25,0,0], [1,4,10,0], [1,4,0,5]])
    
mean = np.zeros((N_test,1))
    
y_test = np.random.multivariate_normal(mu_test, computeK(x_test, THETAS[5]), N_test)    



print y_test.shape

pp.plot(x_test,y_test[0])
pp.plot(x_test,y_test[1])
pp.plot(x_test,y_test[2])
pp.plot(x_test,y_test[3])
pp.plot(x_test,y_test[4])
pp.plot(x_test,y_test[5])
#pp.plot(x_test,y_test)

pp.show()



