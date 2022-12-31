import numpy as np

from commons import hinge_loss_derivative, projection_l1_ball

def prox_svrg(
    train_data_path: str="../data/train_data.npy", train_label_path: str="../data/train_labels.npy", 
    test_data_path: str="../data/test_data.npy", test_label_path: str="../data/test_labels.npy", 
    z: float=100, m: int=140, n_epochs: int=70
):
    """
    Proximal-Stochastic Variance Reduction Gradient

    Params
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
    x_mean = np.zeros(shape=(d, 1))

    for n in range(n_epochs):
        