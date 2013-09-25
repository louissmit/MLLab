from methods import *


(x_train, t_train), (x_valid, t_valid), (x_test, t_test) = load_mnist()

plot_digits(x_train[0:8], numcols=4)
