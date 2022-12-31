import numpy as np
from scipy.optimize import minimize

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
    x_sorted = np.sort(x, axis=0)[::-1]

    # Find the breakpoint
    d = 1
    while d < x.shape[0]:
        sum_ = (np.sum(x_sorted[:d]) - 1) / d
        if sum_ > x_sorted[d]:
            break
        d += 1
    
    theta = (1 / d) * (np.sum(x_sorted[:d]) - 1)
    
    return soft_threshold(x, theta)

def projection_simplex_weighted_norm(x: np.ndarray, D: np.ndarray):
    """
    Function to project a point onto the unit simplew with respect to
    the weighted norm D

    Params:
    x: the point of shape (d x 1) to project
    D: the diagonal weight matrix of shape (d x d)
    """
    # Check if x is already in the simplex
    if np.sum(x) == 1:
        return x
    
    # Sort the coordinate of Dx
    d_x = D * x
    ind = d_x.argsort(axis=0)[::-1]
    d_x_sorted = d_x[ind]
    x_sorted = x[ind]

    # calculate the diagonal of the inverse of D
    d_inversed = (1 / D)
    d_inversed_sorted = d_inversed[ind]

    # Find the breakpoint
    d = 1
    while d < x.shape[0]:
        sum_ =  (np.sum(x_sorted[:d]) - 1) / np.sum(d_inversed_sorted[:d])
        if sum_ > d_x_sorted[d]:
            break        
        d += 1

    theta = 1 / (np.sum(d_inversed_sorted[:d])) * (np.sum(x_sorted[:d]) - 1)

    return d_inversed * soft_threshold(d_x, theta)

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
    # Assert that x is in the simpelx
    # print(np.sum(x_))
    
    return np.sign(x) * x_ * z

def projection_l1_ball_weighted_norm(x: np.ndarray, D: np.ndarray, z: float):
    """
    Function to project a point into the l1 ball of radius z

    Params:
    x: the point of shape (d x 1)
    D: the diagonal weight matrix
    z: radius of the ball
    """
    # Check if x is in the ball or not
    if np.linalg.norm(x, ord=1) <= z:
        return x

    # Compute the projection onto the simplex
    x_ = projection_simplex_weighted_norm(np.abs(x) / z, D)
    # Assert that x is in the simplex
    # print(np.sum(x_))
    
    return np.sign(x) * x_ * z

def general_projection_l1_ball(y: np.ndarray, D: np.ndarray, z: float):
    def obj_func(x, y, D):
        return 0.5 * np.dot(np.dot((x - y).T, D), (x - y))[0, 0]

    cons = ({
        "type": "ineq", "fun": lambda x: z - np.sum(np.abs(x)) 
    })

    result = minimize(
        obj_func,
        np.ones(shape=y.shape) / y.shape[0],
        (y, D),
        constraints=cons
    )

    return result.x[:, None]

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