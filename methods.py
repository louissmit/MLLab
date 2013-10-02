#%pylab inline
import gzip, cPickle
import matplotlib.pyplot as plt
import numpy as np 


def load_mnist():
	f = gzip.open('mnist.pkl.gz', 'rb')
	data = cPickle.load(f)
	f.close()
	return data

def plot_digits(data, numcols, shape=(28,28)):
    numdigits = data.shape[0]
    numrows = int(numdigits/numcols)
    for i in range(numdigits):
        plt.subplot(numrows, numcols, i)
        plt.axis('off')
        plt.imshow(data[i].reshape(shape), interpolation='nearest', cmap='Greys')
    plt.show()


def logreg_gradient(x, t, W, b):
    M,J=W.shape
    N=x.shape
    grad_b = np.zeros(J)
    grad_W = np.zeros((M,J))
    # compute Z
    Z = sum(np.exp(np.dot(W.T, x) + b))
    for j in xrange(J):
        # compute grad_b
        grad_b[j]=-np.exp(np.dot(W[:,j].T, x) + b[j])
    grad_b = grad_b / Z
    grad_b[t]+=1

    for j in xrange(J):
        # compute grad_W
        grad_W[:,j]= x * grad_b[j]
        
    return grad_b, grad_W

def sgd_iter(x_train, t_train, W, b):
    M,J=W.shape
    N,M=x_train.shape
    eta = 1E-4 #learning_rate
    indices = np.arange(N,  dtype = int)
    np.random.shuffle(indices)

    # stochatic gradient decent
    for n in indices:
        grad_b, grad_W = logreg_gradient(x_train[n,:],t_train[n],W,b)
        b+= eta*grad_b 
        W+= eta*grad_W
    return W,b
