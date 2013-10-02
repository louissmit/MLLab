from methods import *
import matplotlib.pyplot as plt

# load data
(x_train, t_train), (x_valid, t_valid), (x_test, t_test) = load_mnist()
# plot_digits(x_train[0:8], numcols=4)


# some iterations
def showProb():
    #initilaize W and b
    N, M = x_train.shape
    J = 10
    b = np.zeros(J)
    W = np.zeros((M,J))

    iterations = 10
    res = np.zeros(iterations)
    for i in xrange(iterations):
        W, b = sgd_iter(x_train, t_train, W, b)
        Z = 0
        logq = 0
        lntrainp = 0
        for n in xrange(N):
            lnq = np.dot(W[:,t_train[n]].T, x_train[n]) + b[t_train[n]]
            lnZ = np.log(sum(np.exp(np.dot(W.T, x_train[n]) + b)))

            # vallnq = np.dot(W[:,t_valid[n]].T, x_valid[:, n]) + b[t_valid[n]]
            # vallnZ = np.log(sum(np.exp(np.dot(W.T, x_valid[:, n]) + b)))

            lntrainp += lnq - lnZ
            # lnvalp += vallnq - vallnZ
        res[i] = lntrainp

    # plt.plot(valprob, label='validation prob.'))
    plt.plot(res, label='train prob.')
    plt.legend()
    plt.show()


def visualizeW():
    #initilaize W and b
    N, M = x_train.shape
    J = 10
    b = np.zeros(J)
    W = np.zeros((M,J))

    for i in xrange(2):
        W, b = sgd_iter(x_train, t_train, W, b)

    plot_digits(W, numcols=10)

showProb()
    # prediction with x_test and t_test
    # calculate error_test
    # prediction with x_valid and t_valid
    # calculate error_valid
    # plot error of training/ validation
    
    # Visulaze W, W[:,j] is one image
