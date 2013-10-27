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
    X_test = np.random.uniform(-1,1,N_test)
    X_test=  np.sort(X_test)
    X_train = np.random.uniform(-1,1,N_train)
    T_train = generate_t(X_train, sigma)
    x_true = np.linspace(-1,1,50)
    y_true = true_mean_function(x_true)

    return X_train, T_train, x_true, y_true, X_test

def true_mean_function( x ):
    return np.sin( pi*x )

def add_noise( y, sigma ):
    return y + sigma*np.random.randn(len(y))

def generate_t( x, sigma ):
    return add_noise( true_mean_function( x), sigma )
    
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

def computeC(K, sigma):
    return K + sigma*sigma * np.identity(K.shape[0])

def show_sample_kernels(X_test, THETAS):
    N_test=len(X_test)
    zero_mean=np.zeros((N_test,))

    pp.figure(figsize=(16, 8))
    for i in xrange(len(THETAS)):
        pp.subplot(2,3,i+1)
        pp.title(r'$\theta$ ='+str(THETAS[i]))
        pp.plot(X_test,np.zeros(X_test.shape),'b--',label='mean')
        K=computeK(X_test, THETAS[i])
        for l in xrange(5):
            y_test = np.random.multivariate_normal(zero_mean,K)
            if not l:
                pp.plot(X_test,y_test,'r', label='GP')
            else:
                pp.plot(X_test,y_test,'r')
            pp.fill_between(X_test,y_test-np.sqrt(np.diag(K)),y_test+np.sqrt(np.diag(K)), alpha=0.05,facecolor='red')
        pp.legend()


#_________________________________________________________________________
# Predictive distribution

def gp_predictive_distribution(X_train, T_train, X_test, theta, sigma, C = None):
    N_train = len(X_train)
    N_test  = len(X_test)
    mu  = np.zeros(N_test)
    var = np.zeros(N_test)
    if not C:
        K = computeK(X_train, theta)
        #C = K + 0.01 * np.identity(N_train) #sigma
        C = computeC(K, sigma)
    Cinv=np.linalg.inv(C)
    
    k = np.empty(N_train)
    for n in xrange(N_test):
        c = k_n_m(X_test[n], X_test[n],theta) + sigma*sigma    
        for i in xrange(N_train):
            k[i] = k_n_m(X_test[n], X_train[i],theta)
        mu[n] = np.dot(np.dot(k.T, Cinv), T_train)
        var[n] = c - np.dot(np.dot(k.T, Cinv), k)
    return mu, var

def gp_log_likelihood( X_train, T_train, theta, C = None, invC = None ):
    N_train = len(X_train)
    if not C:
        K = computeK(X_train, theta)
        #C = K + 0.01 * np.identity(N_train) #sigma
         C = computeC(K, sigma)
    if not invC:
        Cinv=np.linalg.inv(C)

    logLikelihood=-0.5 *np.log(np.linalg.det(C))-0.5*np.dot(np.dot(T_train.T,Cinv),T_train)-N_train/2*np.log(2*pi)# possible errors: det()=determinate, log(), pi
    return logLikelihood, C, Cinv

def gp_plot(X_train, T_train,X_true, Y_true, X_test, beta, THETAS, sigma):
    
    N_test=len(X_test)

    pp.figure(figsize=(16, 8))
    for i in xrange(len(THETAS)):
        pp.subplot(2,3,i+1)

        pp.title(r'$\theta$='+str(THETAS[i]))
        mu, var= gp_predictive_distribution(X_train, T_train, X_test, THETAS[i], sigma)
        
        # separate model and data stddevs.
        std_total = np.sqrt(var)                       # includes all uncertainty, model and target noise 
        std_model = np.sqrt( std_total**2 - 1.0/beta ) # remove data noise to get model uncertainty in stddev
        std_combo = std_model + np.sqrt( 1.0/beta )    # add stddev (note: not the same as full)

        pp.plot(X_train, T_train,'bo', label='training set')
        pp.plot(X_true,Y_true, label='generator')
        pp.plot(X_test,mu,'r--',label='GP posterior')
        pp.fill_between(X_test,mu-2*np.sqrt(var),mu+2*np.sqrt(var), alpha=0.15,facecolor='red')
        #pp.fill_between(X_test,mu-2*std_combo,mu+2*std_combo, alpha=0.15,facecolor='red')
        #pp.fill_between(X_test,mu-2*std_model,mu+2*std_model, alpha=0.15,facecolor='blue')
        pp.ylim(-2,2)
        pp.xlim(-1,1)
        pp.legend(loc=4)
#_________________________________________________________________________
# Learning the hyperparameters


# 3.2 - Performs the grid-search


def similar(a,b, epsilon=0.0000001):
    """ Returns 1 if both numbers are the same numerically at under an epsilon distance. """
    if abs(a-b) < epsilon:
        return 1
    return 0

def grid_search_validation(search_space, search):
    """ Prints a warning in case any of the parameters is on the limit of the search space. """
    print ""
    for i in xrange(len(search)):
        if similar(search[i],max(search_space[i])):  
            print "The parameter {} (with value {}) equals the maximum of it's search space in the grid-search!!".format(i,search[i])
        #if similar(search[i],min(search_space[i])):
        #    print "The parameter {} (with value {}) equals the minimum of it's search space in the grid-search.".format(i,search[i])

def print_log_likelihood_result (result): 
    print " Log-Likelihood:",result[0], "   Thetas:", result[1]

def gen_theta_combinations(thetas):
    """ Generates all the combinations of thetas from a configuration list (search space). """
    theta_combinations = [] 
    for t0 in thetas[0]:
        for t1 in thetas[1]:
            for t2 in thetas[2]:
                for t3 in thetas[3]:
                        theta_combinations.append( (t0,t1,t2,t3) )
    
    return theta_combinations

def grid_search(X_train, T_train, sigma, theta_search_space):
    """ Performs a grid search on the theta-space. The theta-search-space must be a list of lists of the form:
    [[theta0-search-elements], [theta1-search-elements], [theta2-search-elements], [theta3-search-elements]].
    
    Returns the best and the worst results:
    (best, worst)
    
    Note that each of these is in the form:     (likelihood, (theta0, theta1, theta2, theta3))
    
    So if you want to use the theta values from the best result you would do:
    
    best, worst = grid_search(...)
    best_likelihood = best[0]
    best_thetas = best[1]
    """ 
    
    print " --------------------- Grid-search --------------------- "

    theta_combinations = gen_theta_combinations(theta_search_space)

    likelihood_results = []
    for thetas in theta_combinations:
        likelihood_result, C, Cinv = gp_log_likelihood( X_train, T_train, thetas)
        likelihood_results.append( (likelihood_result,thetas) )
        
    likelihood_results = sorted(likelihood_results, key=lambda likelihood_result: likelihood_result[0])

    # This is usually very big but it is asked, so we truncate the printing.
    l = len(likelihood_results)
    if l > 20:
        i = 0
        while i < 15:
            print_log_likelihood_result(likelihood_results[i])
            i += 1
        print " ... the full output would be too large, truncating ... "
        i = l - 15
        while i < l:
            print_log_likelihood_result(likelihood_results[i])
            i += 1

    
    print "\nGrid-search Best result:"
    print_log_likelihood_result(likelihood_results[-1])
    print "Grid-search Worst result:"
    print_log_likelihood_result(likelihood_results[0])
    
    # Issues warnings if we are at the upper boundary of the search space in any variable 
    grid_search_validation(theta_search_space, likelihood_results[-1][1])    
    
    return (likelihood_results[-1], likelihood_results[0])



# 3.7 Bonus

def grad_lnp_function(Cinv, gradC, T_train): # Auxiliary function
    return 0.5 * ( np.dot(np.dot(np.dot(T_train.T,Cinv), gradC), T_train) - np.trace(np.dot(Cinv,gradC))  )

def grad_lnp(thetas, X_train, T_train,Cinv):
    grad_lnp=np.zeros((4,1))
    N_train=len(X_train)
    gradC=np.zeros((N_train,N_train))

    # Theta_0
    for n in xrange(N_train):
        for m in xrange(N_train):
            gradC[n,m]= np.exp((-np.log(thetas[1])/2.0) * abs2(X_train[n]-X_train[m])) * thetas[0]
    #grad_lnp[0]=-0.5* np.trace(np.dot(Cinv,gradC))+0.5* np.dot(np.dot(np.dot(T_train.T,Cinv), gradC), T_train)
    grad_lnp[0] = grad_lnp_function(Cinv, gradC, T_train)
   
    # Theta_1
    for n in xrange(N_train):
        for m in xrange(N_train):
            norm = abs2(X_train[n]-X_train[m])
            first_term =(-np.log(thetas[0])/2.0) * norm 
            second_term = np.exp((-np.log(thetas[1])/2.0) * norm)
            gradC[n,m] = first_term *  second_term  * thetas[1]
    #grad_lnp[1]=-0.5* np.trace(np.dot(Cinv,gradC))+0.5* np.dot(np.dot(np.dot(T_train.T,Cinv), gradC), T_train)
    grad_lnp[1] = grad_lnp_function(Cinv, gradC, T_train)
   
    # Theta_2
    for n in xrange(N_train):
        for m in xrange(N_train):
            gradC[n,m]= thetas[2]
    #grad_lnp[2]=-0.5* np.trace(np.dot(Cinv,gradC))+0.5* np.dot(np.dot(np.dot(T_train.T,Cinv), gradC), T_train)
    grad_lnp[2] = grad_lnp_function(Cinv, gradC, T_train)
   
    # Theta_3
    for n in xrange(N_train):
        for m in xrange(N_train):
            gradC[n,m]= X_train[n]*X_train[m]*thetas[3]
    #grad_lnp[3]=-0.5* np.trace(np.dot(Cinv,gradC))+0.5* np.dot(np.dot(np.dot(T_train.T,Cinv), gradC), T_train)
    grad_lnp[3] = grad_lnp_function(Cinv, gradC, T_train)
    
    return grad_lnp

#_________________________________________________________________________
