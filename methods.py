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

    Z = sum(np.exp(np.dot(W.T, x) + b))

    grad_b = -np.exp(np.dot(W.T, x) + b) / Z
    grad_b[t]+=1

    xtile = np.tile(x, (J, 1)).T
    grad_W = np.dot(xtile, np.diag(grad_b))

    return grad_b, grad_W

def sgd_iter(x_train, t_train, W, b):
    M,J=W.shape
    N,M=x_train.shape
    eta = 1E-4 #learning_rate
    #indices = np.arange(N,  dtype = int)
    #np.random.shuffle(indices)

    # stochatic gradient decent
    for n in xrange(N):#indices:
        grad_b, grad_W = logreg_gradient(x_train[n],t_train[n],W,b)
        b+= eta*grad_b 
        W+= eta*grad_W
    return W,b
