import numpy as np
from math import sqrt

from commons import accuracy, hinge_loss_derivative

def sreg(
    train_data_path: str="../data/train_data.npy", train_label_path: str="../data/train_labels.npy", 
    test_data_path: str="../data/test_data.npy", test_label_path: str="../data/test_labels.npy", 
    z: float=1, n_epochs: int=10000
):
    """
    Stochastic Randomized Exponentiated Gradient

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
    w = np.ones(shape=(2 * d, 1)) * (1 / (2 * d))

    m = x
    accuracies.append(accuracy(m, a_test, b_test))

    for n in range(n_epochs):
        index = np.random.randint(low=0, high=n_samples, size=(1,))
        direction = np.random.randint(low=0, high=d)

        if n == 0:
            lr = 0
        else:
            lr = 1 / sqrt(d * n)
        
        grad = hinge_loss_derivative(x, a_train[index], b_train[index])
        w[direction] = np.exp(-lr * d * grad[direction]) * w[direction]
        w[direction + d] = np.exp(lr * d * grad[direction]) * w[direction + d]
        w = w / np.sum(w)
        x = z * (w[:d] - w[d:])

        m = (n * m + x) / (n + 1)

        if (n + 1) % (n_epochs // 100) == 0:
            acc = accuracy(m, a_test, b_test)
            accuracies.append(acc)

    return m, accuracies

def sbeg(
    train_data_path: str="../data/train_data.npy", train_label_path: str="../data/train_labels.npy", 
    test_data_path: str="../data/test_data.npy", test_label_path: str="../data/test_labels.npy", 
    z: float=1, n_epochs: int=10000
):
    """
    Stochastic Bandit Exponentiated Gradient

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
    w = np.ones(shape=(2 * d, 1)) * (1 / (2 * d))
    w_ = np.ones(shape=(2 * d, 1)) * (1 / (2 * d))

    m = x
    accuracies.append(accuracy(m, a_test, b_test))

    for n in range(n_epochs):
        index = np.random.randint(low=0, high=n_samples, size=(1,))
        action = np.random.choice(2 * d, p=np.squeeze(w))
        action_index = action * (action < d) + (action - d) * (action >= d)
        s = 2 * (action < d) - 1

        if n == 0:
            lr = 0
        else:
            lr = 1 / sqrt(d * n)
        gamma = min(1, d * lr)

        grad = hinge_loss_derivative(x, a_train[index], b_train[index])
        w_[action] = np.exp(-lr * s * grad[action_index] / w[action]) * w_[action]
        w_ = w_ / np.sum(w_)
        w = (1 - gamma) * w_ + gamma / (2 * d)
        x = z * (w_[:d] - w_[d:])

        m = (n * m + x) / (n + 1)

        if (n + 1) % (n_epochs // 100) == 0:
            acc = accuracy(m, a_test, b_test)
            accuracies.append(acc)

    return m, accuracies