#_________________________________________________________________________
# Libraries
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
    return thetas[0]*np.exp(-thetas[1]/2.0 * abs2(xn-xm)) + thetas[2] + thetas[3]*np.dot(xn.T, xm)

def computeK(X, thetas):
    N = X.shape[0]
    K = np.zeros((N,N))    

    for n in xrange(N):
        for m in xrange(N):
            K[n,m] = k_n_m(X[n], X[m], thetas)

    return K

def show_sample_kernels():
    THETAS = np.array([[1,4,0,0], [9,4,0,0], [1,64,0,0], [1,0.25,0,0], [1,4,10,0], [1,4,0,5]])
    mean = np.zeros((N_test,1))
    y_test = np.random.multivariate_normal(mu_test, computeK(x_test, THETAS[5]), N_test)    

    pp.plot(x_test,y_test[0])
    pp.plot(x_test,y_test[1])
    pp.plot(x_test,y_test[2])
    pp.plot(x_test,y_test[3])
    pp.plot(x_test,y_test[4])
    pp.plot(x_test,y_test[5])
    pp.show()

#_________________________________________________________________________
# Predictive distribution

def gp_predictive_distribution(X_train, T_train, X_test, theta, C = None):
    D, N_train = X_train.shape
    D, N_test =X_test.shape
    mu = np.zeros(N_test)
    var = np.zeros(N_test)
    if not C:
        K = computeK(X_train, theta)
        C = K + 0.01 * np.identity(N_train) #sigma?
    Cinv=np.inverse(C)
    
    k = np.empty(N_train)
    for n in xrange(N_test):
        c = k_n_m(X_test[i], X_test[i]) + 0.01    
        for i in xrange(N_train):
            k[i] = k_n_m(X_test[n], X_train[i])
        mu[n] = np.dot(np.dot(k.T, Cinv), T_train)
        var[n] = c - np.dot(np.dot(k.T, Cinv), k)
return mu, var

def gp_log_likelihood( X_train, T_train, theta, C = None, invC = None ):
    D, N_train = X_train.shape
    if not C:
        K = computeK(X_train, theta)
        C = K + 0.01 * np.identity(N_train) #sigma?
    if not invC:
        Cinv=np.inverse(C)

    logLikelihood=-0.5 log(det(C))-0.5*np.dot(np.dot(T_train.T,Cinv),t)-N_train/2*log(2*pi)# possible errors: det()=determinate, log(), pi
    return logLikelihood


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
# Learning the hyperparameters
K = computeK(X_train, theta)
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

sort-dictionary-thingy(...)
plot-results-KAREN :P:P(...)

#_________________________________________________________________________
