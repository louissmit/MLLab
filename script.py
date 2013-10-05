from methods import *
import matplotlib.pyplot as plt

# load data
(x_train, t_train), (x_valid, t_valid), (x_test, t_test) = load_mnist()
# plot_digits(x_train[0:8], numcols=4)




# some iterations
def showProb():
    #initilaize W and b
    N, M = x_train.shape
    K, L = x_valid.shape
    J = 10
    b = np.zeros(J)
    W = np.zeros((M,J))

    iterations = 7
    trainres = np.zeros(iterations)
    valres = np.zeros(iterations)
    for i in xrange(iterations):
        W, b = sgd_iter(x_train, t_train, W, b)
        lntrainp = 0
        lnvalidp = 0
        for n in xrange(N):
            lnq = np.dot(W[:,t_train[n]].T, x_train[n]) + b[t_train[n]]
            lnZ = np.log(sum(np.exp(np.dot(W.T, x_train[n]) + b)))

            if(n < K):
                vallnq = np.dot(W[:,t_valid[n]].T, x_valid[n]) + b[t_valid[n]]
                vallnZ = np.log(sum(np.exp(np.dot(W.T, x_valid[n]) + b)))

            lntrainp += lnq - lnZ
            lnvalidp += vallnq - vallnZ
        trainres[i] = lntrainp / N
        valres[i] = lnvalidp / K

    plt.plot(trainres, label='train prob.')
    plt.plot(valres, label='validation prob.')
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

    plot_digits(W.T, numcols=5)

def getHardestOrEasiest(hardest):
    #initilaize W and b
    N, M = x_valid.shape
    J = 10
    b = np.zeros(J)
    W = np.zeros((M,J))

    iterations = 8
    valres = np.zeros(iterations)

    for i in xrange(iterations):
        W, b = sgd_iter(x_train, t_train, W, b)

    lnvalidp = {}
    for n in xrange(N):
        lnq = np.dot(W[:,t_valid[n]].T, x_valid[n]) + b[t_valid[n]]
        lnZ = np.log(sum(np.exp(np.dot(W.T, x_valid[n]) + b)))

        lnvalidp[n] = lnq - lnZ

    res = sorted(lnvalidp, key=lnvalidp.get)
    R = np.zeros((8, M))
    for x in xrange(8):
        if hardest:
            R[x] = x_valid[res[x]].T
        else:
            R[x] = x_valid[res[-x]].T

    plot_digits(R, numcols=4)


# visualizeW()
# showProb()
getHardestOrEasiest(False)


    # prediction with x_test and t_test
    # calculate error_test
    # prediction with x_valid and t_valid
    # calculate error_valid
    # plot error of training/ validation
    
    # Visulaze W, W[:,j] is one image
