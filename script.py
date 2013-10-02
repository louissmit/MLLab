from methods import *
import matplotlib.pyplot as plt

# load data
(x_train, t_train), (x_valid, t_valid), (x_test, t_test) = load_mnist()
# plot_digits(x_train[0:8], numcols=4)

#initilaize W and b
N, M = x_train.shape
J = 10
b = np.zeros(J)
W = np.zeros((M,J))

# some iterations
for i in xrange(5):
    W, b = sgd_iter(x_train, t_train, W, b)
    Z = 0
    logq = np.zeros(J)
    for j in xrange(J):
        logq[j] = np.dot(W[:,j].T, x_train[i]) + b[j]
        Z += np.exp(logq[j])
    res = np.exp(logq - np.log(Z))
    plt.plot(res, label="i = " + str(i))
plt.legend()
plt.show()


    # prediction with x_test and t_test
    # calculate error_test
    # prediction with x_valid and t_valid
    # calculate error_valid
    # plot error of training/ validation
    
    # Visulaze W, W[:,j] is one image
