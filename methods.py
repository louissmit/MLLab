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
    logq = np.exp(np.dot(W.T, x) + b)

    grad_b = - logq / sum(logq) 
    grad_b[t] += 1

    grad_W = np.outer(x, grad_b.T)

    return grad_b, grad_W


def sgd_iter(x_train, t_train, W, b):
    N,M=x_train.shape
    eta = 1E-4 #learning_rate
    indices = np.arange(N,  dtype = int)
    np.random.shuffle(indices)

    # stochatic gradient decent
    for n in indices:
        grad_b, grad_W = logreg_gradient(x_train[n],t_train[n],W,b)
        b+= eta*grad_b 
        W+= eta*grad_W
    return W,b

def percept_gradient(x, t, W, V, b, a):
    # feed forward
    h = 1 / (1 + np.exp(np.dot(V.T, x) + a))
    # first backward pass
    logq = np.exp(np.dot(W.T, h) + b)
    grad_b = - logq / sum(logq) 
    grad_b[t] += 1
    grad_W = np.outer(h, grad_b.T)

    # second backward pass
    print(W.shape)
    delta_h = W[:,t] - sum(W * np.exp(np.dot(W.T, h) + b)) / sum(np.exp(np.dot(W.T, h) + b))
    grad_a = delta_h * h * (1 - h)
    grad_V = np.outer(x, grad_a.T)

    return grad_b, grad_W, grad_a, grad_V


def percept_sgd_iter(x_train, t_train, W, V, b, a):
    N,M=x_train.shape
    eta = 1E-4 #learning_rate
    indices = np.arange(N,  dtype = int)
    np.random.shuffle(indices)

    # stochatic gradient decent
    for n in indices:
        grad_b, grad_W, grad_a, grad_V = percept_gradient(x_train[n],t_train[n],W,V,b,a)
        b+= eta*grad_b 
        a+= eta*grad_a 
        W+= eta*grad_W
        V+= eta*grad_V
    return W,b
