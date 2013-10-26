#_________________________________________________________________________
# Libraries

from math import pi as pi
import numpy as np
import matplotlib.pyplot as pp

#_________________________________________________________________________
# Data Generator
def data_gen(N_train,N_test,sigma=None):
    if not sigma:
        sigma = 0.1
    beta  = 1.0 / pow(sigma,2) # this is the beta used in Bishop Eqn. 6.59
    X_test = np.linspace(-1,1,N_test)
    X_train = np.linspace(-1,1,N_train)
    T_train = generate_t( X_train, sigma)
    y_train = true_mean_function( X_train)

    return X_train, T_train, X_test, y_train

def true_mean_function( x ):
    return np.sin( 2*pi*(x+1) )

def add_noise( y, sigma ):
    return y + sigma*np.random.randn(len(y))

def generate_t( x, sigma ):
    return add_noise( true_mean_function( x), sigma )
    


#pp.plot( x_test, y_test, 'b-', lw=2)
#pp.plot( x_test, t_test, 'go')

#_________________________________________________________________________
# Sampling from the Gaussian process prior

def abs2(vec):
    """ Norm of the vector squared """
    return np.sum(vec*vec)

def k_n_m(xn, xm, thetas):
    """ Returns the Kernel element Knm """
    
    # This would be in a multi-dimensional case
    #return thetas[0]*np.exp(-thetas[1]/2.0 * abs2(xn-xm)) + thetas[2] + thetas[3]*np.dot(xn.T, xm)
    
    # Faster 1D implementation
    dnm = xn-xm
    return thetas[0]*np.exp(-0.5*thetas[1] * (dnm*dnm)) + thetas[2] + thetas[3]*(xn*xm)

def computeK(X, thetas):
    N = X.shape[0]
    K = np.zeros((N,N))    

    for n in xrange(N):
        for m in xrange(N):
            K[n][m] = k_n_m(X[n], X[m], thetas)

    return K

def computeC(K,sigma):
    """ Computes the matrix C. """
    # C = K + I_N / beta = K + sigma * I_N  
    return K + sigma * np.identity(K.shape[0])

def show_sample_kernels(N_test, X_test, mu_test, thetaset):
    """ Still need to make the plot in a grid and then you can make thetaset choice disappear"""
    
    #TODO: Remove the parameter thetaset and use a subfig that iterates through all the thetasets. After that, finish the plotting with the requiremnents in the assignment.    
    
    THETAS = np.array([[1,4,0,0], [9,4,0,0], [1,64,0,0], [1,0.25,0,0], [1,4,10,0], [1,4,0,5]])
    mean = np.zeros((N_test,1))
    
    y_test = np.random.multivariate_normal(mu_test, computeK(X_test, THETAS[thetaset]), N_test)    

    
    pp.plot(X_test,y_test[0])
    pp.plot(X_test,y_test[1])
    pp.plot(X_test,y_test[2])
    pp.plot(X_test,y_test[3])
    pp.plot(X_test,y_test[4])
    pp.plot(X_test,y_test[5])
    pp.show()

#_________________________________________________________________________
# Predictive distribution

def gp_predictive_distribution(X_train, T_train, X_test, theta, sigma, C = None):
    N_train = X_train.shape[0]
    N_test = X_test.shape[0]
    mu = np.zeros(N_test)
    var = np.zeros(N_test)
    
    if not C:
        K = computeK(X_train, theta)
        C = computeC(K, sigma)
    Cinv = np.linalg.inv(C)
    
    # For each of the test points we calculate C_{N+1} and parameters mu & var.
    k = np.empty(N_train)
    for n in xrange(N_test):
        c = k_n_m(X_test[n], X_test[n], theta) + (1/float(sigma))  # k(x_n+1,x_n+1) + beta
        
        for i in xrange(N_train):
            k[i] = k_n_m(X_test[n], X_train[i], theta)
            
        mu[n] = np.dot(np.dot(k.T, Cinv), T_train)
        var[n] = c - np.dot(np.dot(k.T, Cinv), k)
    return mu, var

def gp_log_likelihood( X_train, T_train, theta, sigma, C = None, invC = None ):
    """ Returns the log-likelihood as well as C and C^-1 which can be reused. """
    N_train = X_train.shape[0]
    if not C:
        K = computeK(X_train, theta)
        C = computeC(K, sigma)
    if not invC:
        Cinv=np.linalg.inv(C)

    logLikelihood=-0.5*(np.log(np.linalg.det(C)) + np.dot(np.dot(T_train.T,Cinv),T_train) + N_train*np.log(2*pi))
    return logLikelihood, C, Cinv


def gp_plot( x_test, y_test, mu_test, var_test, x_train, t_train, theta, beta ):
    # x_test: 
    # y_test:   the true function at x_test
    # mu_test:  predictive mean at x_test
    # var_test: predictive covariance at x_test 
    # t_train:  the training values
    # theta:    the kernel parameters
    # beta:     the precision (known)
    
    # the reason for the manipulation is to allow plots separating model and data stddevs.
    std_total = np.sqrt(np.diag(var_test))         # includes all uncertainty, model and target noise 
    std_model = np.sqrt( std_total**2 - 1.0/beta ) # remove data noise to get model uncertainty in stddev
    std_combo = std_model + np.sqrt( 1.0/beta )    # add stddev (note: not the same as full)
    
    pp.plot( x_test, y_test, 'b', lw=3)
    pp.plot( x_test, mu_test, 'k--', lw=2 )
    pp.fill_between( x_test, mu_test+2*std_combo,mu_test-2*std_combo, color='k', alpha=0.25 )
    pp.fill_between( x_test, mu_test+2*std_model,mu_test-2*std_model, color='r', alpha=0.25 )
    pp.plot( x_train, t_train, 'ro', ms=10 )

#_________________________________________________________________________



def gen_theta_combinations(thetas):
    """ Generates all the combinations of thetas from a configuration list (search space). """
    theta_combinations = [] 
    for t0 in thetas[0]:
        for t1 in thetas[1]:
            for t2 in thetas[2]:
                for t3 in thetas[3]:
                        theta_combinations.append( (t0,t1,t2,t3) )
    
    return theta_combinations