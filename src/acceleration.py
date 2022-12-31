import numpy as np

from commons import accuracy, general_projection_l1_ball, hinge_loss_derivative

def ons(
    train_data_path: str="../data/train_data.npy", train_label_path: str="../data/train_labels.npy", 
    test_data_path: str="../data/test_data.npy", test_label_path: str="../data/test_labels.npy", 
    z: float=100, lambda_: float=1/3, gamma: float=1/8, n_epochs: int=10000
):
    """
    Online Newton Step
    
    Params:
    train_data_path, train_label_path: path to train data and train label file
    test_data_path, test_label_path: path to test data and test label file
    z: l1 ball radius
    lambda_: regularization coef
    gamma: gamma coef
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
    A = np.eye(d) * (1 / gamma ** 2)
    A_inversed = np.eye(d) * (1 / gamma ** 2)

    m = x
    accuracies.append(accuracy(m, a_test, b_test))

    for n in range(n_epochs):
        index = np.random.randint(low=0, high=n_samples, size=(1,))

        grad = hinge_loss_derivative(x, a_train[index], b_train[index])
        delta = grad + lambda_ * x
        A = A + np.dot(delta, delta.T)
        A_inversed = A_inversed - (np.dot(np.dot(np.dot(A_inversed, delta), delta.T), A_inversed)) / \
                                  (1 + np.dot(np.dot(delta.T, A_inversed), delta))
        y = x - (1 / gamma) * np.dot(A_inversed, delta)
        x = general_projection_l1_ball(y, A, z)

        m = (n * m + x) / (n + 1)

        if (n + 1) % (n_epochs // 100) == 0:
            acc = accuracy(m, a_test, b_test)
            accuracies.append(acc)

    return m, accuracies