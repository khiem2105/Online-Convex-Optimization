import numpy as np

from commons import hinge_loss_derivative, projection_l1_ball, accuracy

def prox_svrg(
    lr: float, train_data_path: str="../data/train_data.npy", train_label_path: str="../data/train_labels.npy", 
    test_data_path: str="../data/test_data.npy", test_label_path: str="../data/test_labels.npy", 
    z: float=100, m: int=140, n_epochs: int=70
):
    """
    Proximal-Stochastic Variance Reduction Gradient

    Params
    lr: learning rate
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

    x_mean = np.zeros(shape=(d, 1))

    for n in range(n_epochs):
        full_grad = hinge_loss_derivative(x_mean, a_train, b_train)
        
        x = x_mean
        
        for i in range(m):
            index = np.random.randint(low=0, high=n_samples, size=(1,))
            
            v = hinge_loss_derivative(x, a_train[index], b_train[index]) -\
                hinge_loss_derivative(x_mean, a_train[index], b_train[index]) + full_grad
            x = projection_l1_ball(x - lr * v, z)

            x_mean = (i * x_mean + x) / (i + 1)
        
        acc = accuracy(x_mean, a_test, b_test)
        accuracies.append(acc)

    return x_mean, accuracies
        
