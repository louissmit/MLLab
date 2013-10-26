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

def show_sample_kernels(X_test):
    THETAS = np.array([[1,4,0,0], [9,4,0,0], [1,64,0,0], [1,0.25,0,0], [1,4,10,0], [1,4,0,5]])
    N_test=len(X_test)
    zero_mean=np.zeros((N_test,))


    for i in xrange(1):#xrange(len(THETAS)):
        pp.title('theta='+str(THETAS[i]))
        K=computeK(X_test, THETAS[i])
        y_test = np.random.multivariate_normal(zero_mean,K)
        pp.plot(X_test,np.zeros(X_test.shape),'b--',label='expected mean')
        pp.plot(X_test,y_test,'r', label='GPP')
        pp.fill_between(X_test,y_test-2*np.diag(K)[1],y_test+2*np.diag(K)[1], alpha=0.15,facecolor='red')
        pp.legend()

#_________________________________________________________________________
# Predictive distribution

def gp_predictive_distribution(X_train, T_train, X_test, theta, C = None):
    N_train = len(X_train)
    N_test  = len(X_test)
    mu  = np.zeros(N_test)
    var = np.zeros(N_test)
    if not C:
        K = computeK(X_train, theta)
        C = K + 0.01 * np.identity(N_train) #sigma
    Cinv=np.linalg.inv(C)
    
    k = np.empty(N_train)
    for n in xrange(N_test):
        c = k_n_m(X_test[n], X_test[n],theta) + 0.01    
        for i in xrange(N_train):
            k[i] = k_n_m(X_test[n], X_train[i],theta)
        mu[n] = np.dot(np.dot(k.T, Cinv), T_train)
        var[n] = c - np.dot(np.dot(k.T, Cinv), k)
    return mu, var

def gp_log_likelihood( X_train, T_train, theta, C = None, invC = None ):
    N_train = len(X_train)
    if not C:
        K = computeK(X_train, theta)
        C = K + 0.01 * np.identity(N_train) #sigma?
    if not invC:
        Cinv=np.linalg.inv(C)

    logLikelihood=-0.5 *np.log(np.linalg.det(C))-0.5*np.dot(np.dot(T_train.T,Cinv),T_train)-N_train/2*np.log(2*pi)# possible errors: det()=determinate, log(), pi
    return logLikelihood, C, Cinv


def gp_plot(X_train, T_train,X_true, Y_true, X_test, beta, THETAS):
    
    
    N_test=len(X_test)

    for i in xrange(1):#xrange(len(THETAS)):
        pp.title('theta='+str(THETAS[i]))
        mu, var= gp_predictive_distribution(X_train, T_train, X_test, THETAS[i])
        
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
        pp.legend()
#_________________________________________________________________________
# Learning the hyperparameters

#K = computeK(X_train, theta)
#C = K + 0.01 * np.identity(N_train) 
#Cinv=np.linalg.inv(C)

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
    """ 
    
    print " --------------------- Grid-search --------------------- "

    theta_combinations = gen_theta_combinations(theta_search_space)

    likelihood_results = []
    for thetas in theta_combinations:
        likelihood_result, C, Cinv = gp_log_likelihood( X_train, T_train, thetas)
        likelihood_results.append( (likelihood_result,thetas) )
        
    likelihood_results = sorted(likelihood_results, key=lambda likelihood_result: likelihood_result[0])

    # This is usually very big but it is asked.
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

def grad_lnp(thetas, X_train, T_train,Cinv):
    grad_lnp=np.zeros((4,1))
    N_train=len(X_train)
    gradC=np.zeros((N_train,N_train))

    # Theta_0
    for n in xrange(N_train):
        for m in xrange(N_train):
            gradC[n,m]= np.exp(-thetas[1]/2.0 * abs2(X_train[n]-X_train[m]))*np.exp(thetas[0])
    grad_lnp[0]=-0.5 *np.trace(np.dot(Cinv,gradC))+0.5* np.dot(np.dot(np.dot(T_train.T,Cinv),gradC),T_train)
    # Theta_1
    for n in xrange(N_train):
        for m in xrange(N_train):
            gradC[n,m]= np.exp(-thetas[1]/2.0 * abs2(X_train[n]-X_train[m]))*-thetas[0]/2.0 * abs2(X_train[n]-X_train[m])*np.exp(thetas[1])
    grad_lnp[1]=-0.5* np.trace(np.dot(Cinv,gradC))+0.5* np.dot(np.dot(np.dot(T_train.T,Cinv),gradC),T_train)
    # Theta_2
    for n in xrange(N_train):
        for m in xrange(N_train):
            gradC[n,m]= np.exp(thetas[2])
    grad_lnp[2]=-0.5* np.trace(np.dot(Cinv,gradC))+0.5* np.dot(np.dot(np.dot(T_train.T,Cinv),gradC),T_train)
    # Theta_3
    for n in xrange(N_train):
        for m in xrange(N_train):
            gradC[n,m]= X_train[n]*X_train[m]*np.exp(thetas[3])
    grad_lnp[3]=-0.5* np.trace(np.dot(Cinv,gradC))+0.5* np.dot(np.dot(np.dot(T_train.T,Cinv),gradC),T_train)
    
    return grad_lnp

#_________________________________________________________________________
