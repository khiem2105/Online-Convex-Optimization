import numpy as np
from math import sqrt

from commons import hinge_loss_regularized_derivative, accuracy, projection_l1_ball

def gd(train_data_path: str="../data/train_data.npy", train_label_path: str="../data/train_labels.npy", 
       test_data_path: str="../data/test_data.npy", test_label_path: str="../data/test_labels.npy", 
       lambda_: float=1/3, z: float=1, n_epochs: int=100, projected=False):
    """
    Gradient descent

    Params:
    train_data_path, train_label_path: path to train data and train label file
    test_data_path, test_label_path: path to test data and test label file
    lambda_: regularization coef
    z: l1 ball radius
    n_epochs: max epochs
    projected: whether or not to project into the l1 ball
    """

    a_train = np.load(train_data_path)
    b_train = np.load(train_label_path)

    a_test = np.load(test_data_path)
    b_test = np.load(test_label_path)

    accuracies = []

    x = np.zeros(shape=(a_train.shape[1], 1))

    for n in range(n_epochs):
        if n == 0:
            lr = 0
        else:
            lr = 1 / (lambda_ * n)

        grad = hinge_loss_regularized_derivative(x, a_train, b_train, lambda_)
        x = x - lr * grad
        if projected:
            x = projection_l1_ball(x, z)
        
        acc = accuracy(x, a_test, b_test)
        accuracies.append(acc)

    return x, accuracies

def sgd(
    train_data_path: str="../data/train_data.npy", train_label_path: str="../data/train_labels.npy", 
    test_data_path: str="../data/test_data.npy", test_label_path: str="../data/test_labels.npy", 
    lambda_: float=1/3, z: float=1, n_epochs: int=10000, projected: bool=False, sqrt_lr: bool=False
):
    """
    Stochastic unconstrained gradient descent

    Params:
    train_data_path, train_label_path: path to train data and train label file
    test_data_path, test_label_path: path to test data and test label file
    lambda_: regularization coef
    z: l1 ball radius
    n_epochs: max epochs
    projected: whether or not to project into the l1 ball
    sqrt_lr: whether or not to use lr = 1 / sqrt(t)
    """

    a_train = np.load(train_data_path)
    b_train = np.load(train_label_path)

    a_test = np.load(test_data_path)
    b_test = np.load(test_label_path)

    n_samples, d = a_train.shape

    accuracies = []

    x = np.zeros(shape=(d, 1))
    m = x

    for n in range(n_epochs):
        index = np.random.randint(low=0, high=n_samples, size=(1,))
        
        if n == 0:
            lr = 0
        else:
            if sqrt_lr:
                lr =  1 / sqrt(n)
            else:
                lr = 1 / (lambda_ * n)
        
        grad = hinge_loss_regularized_derivative(x, a_train[index], b_train[index], lambda_)
        x = x - lr * grad
        if projected:
            x = projection_l1_ball(x, z)
        m = (n * m + x) / (n + 1)

        if (n + 1) % (n_epochs // 100) == 0:
            acc = accuracy(m, a_test, b_test)
            accuracies.append(acc)

    return m, accuracies



