import numpy as np

def hinge_loss_regularized_derivative(x: np.ndarray, a: np.ndarray, b: np.ndarray, lambda_: float):
    """
    Function to compute the derivative of the regularized hinge loss

    Params:
    x: parameters of shape (d x 1)
    a: data of shape (n x d)
    b: labels of shape (n x 1)
    lambda_: regularized coef
    """
    return lambda_ * x + np.mean(-b * a * ((1 - b * np.dot(a, x)) >= 0), axis=0, keepdims=True).T

def hinge_loss_derivative(x: np.ndarray, a: np.ndarray, b: np.ndarray):
    """
    Function to compute the derivative of the regularized hinge loss

    Params:
    x: parameters of shape (d x 1)
    a: data of shape (n x d)
    b: labels of shape (n x 1)
    lambda_: regularized coef
    """
    return np.mean(-b * a * ((1 - b * np.dot(a, x)) >= 0), axis=0, keepdims=True).T

def soft_threshold(x: np.ndarray, theta: float):
    """
    Function to compute the soft threshold of a point

    Params:
    x: the point of shape (d x 1)
    theta: the distance to subtract
    """
    return np.array(
        list(
            map(
                lambda x: x if x[0] >= 0 else [0],
                x - theta
            )
        )
    )

def projection_simplex(x: np.ndarray):
    """
    Function to project a point onto the unit simplex

    Param:
    x: the point of shape (d x 1) to project
    """
    # Check if x is already in the simplex
    if np.sum(x) == 1:
        return x

    # Sort the coordinate in descending order
    x_sorted = np.sort(x)

    # Find the breakpoint
    d = 1
    while d < x.shape[0]:
        sum_ = np.sum(x_sorted[:d]) - x_sorted[d]
        if sum_ > 1:
            break
        d += 1
    
    theta = (1 / d) * (np.sum(x_sorted[:d]) - 1)
    
    return soft_threshold(x, theta)

def projection_l1_ball(x: np.ndarray, z: float):
    """
    Function to project a point into the l1 ball of radius z

    Params:
    x: the point of shape (d x 1)
    z: radius of the ball
    """ 
    # Check if x is in the ball or not
    if np.linalg.norm(x, ord=1) <= z:
        return x
    
    # Compute the projection onto the simplex
    x_ = projection_simplex(np.abs(x) / z)
    
    return np.sign(x) * x_ * z

def accuracy(x: np.ndarray, a_test: np.ndarray, b_test: np.ndarray):
    """
    Function to compute the accuracy
    
    Params:
    x: linear SVM param of shape (d x 1)
    a_test: test data of shape (n x d)
    b_test: test label of shape (n x 1)
    """
    b_predicted = np.array(
        list(
            map(
                lambda x: [1] if x[0] >= 0 else [-1],
                np.dot(a_test, x)
            )
        )
    )

    return 1 - np.mean(b_predicted == b_test)