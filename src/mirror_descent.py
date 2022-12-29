import numpy as np
from math import sqrt

from commons import accuracy, projection_l1_ball, hinge_loss_derivative

def smd(
    train_data_path: str="../data/train_data.npy", train_label_path: str="../data/train_labels.npy", 
    test_data_path: str="../data/test_data.npy", test_label_path: str="../data/test_labels.npy", 
    z: float=1, n_epochs: int=10000
):
    """
    Stochastic mirror descent

    Params:

    train_data_path, train_label_path: path to train data and train label file
    test_data_path, test_label_path: path to test data and test label file
    z: l1 ball radius
    n_epochs: max epochs
    """
    a_train = np.load(train_data_path)
    b_train = np.load(train_label_path)

    a_test = np.load(test_data_path)
    b_test = np.load(test_label_path)

    n_samples, d = a_train.shape

    accuracies = []

    x = np.zeros(shape=(d, 1))
    y = np.zeros(shape=(d, 1))

    m = x

    for n in range(n_epochs):
        index = np.random.randint(low=0, high=n_samples, size=(1,))

        if n == 0:
            lr = 0
        else:
            lr = 1 / sqrt(n)

        grad = hinge_loss_derivative(x, a_train[index], b_train[index])
        y = y - lr * grad
        x = projection_l1_ball(y, z)

        m = (n * m + x) / (n + 1)

        if (n + 1) % (n_epochs // 100) == 0:
            acc = accuracy(m, a_test, b_test)
            accuracies.append(acc)

    return m, accuracies

def seg(
    train_data_path: str="../data/train_data.npy", train_label_path: str="../data/train_labels.npy", 
    test_data_path: str="../data/test_data.npy", test_label_path: str="../data/test_labels.npy", 
    z: float=1, n_epochs: int=10000
):
    """
    Stochastic Exponentiated Gradient Descent

    train_data_path, train_label_path: path to train data and train label file
    test_data_path, test_label_path: path to test data and test label file
    z: l1 ball radius
    n_epochs: max epochs
    """

    a_train = np.load(train_data_path)
    b_train = np.load(train_label_path)

    a_test = np.load(test_data_path)
    b_test = np.load(test_label_path)

    n_samples, d = a_train.shape

    accuracies = []

    x = np.zeros(shape=(d, 1))
    theta = np.zeros(shape=(2 * d, 1))
    w = np.full(shape=(2 * d, 1), fill_value=1 / (2 * d))

    m = x
    accuracies.append(accuracy(m, a_test,  b_test))

    for n in range(n_epochs):
        index = np.random.randint(low=0, high=n_samples, size=(1,))

        if n == 0:
            lr = 0
        else:
            lr = 1 / sqrt(n)

        grad = hinge_loss_derivative(x, a_train[index], b_train[index])
        theta[:d] = theta[:d] - lr * grad
        theta[d:] = theta[d:] + lr * grad

        w = np.exp(theta) / np.sum(np.exp(theta))

        x = z * (w[:d] - w[d:])
        m = (n * m + x) / (n + 1)

        if (n + 1) % (n_epochs // 100) == 0:
            acc = accuracy(m, a_test, b_test)
            accuracies.append(acc)

    return m, accuracies