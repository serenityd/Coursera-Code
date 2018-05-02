import numpy as np
from computeCost import computeCost


def gradientDescent(X, y, theta, alpha, num_iters):
    """
     Performs gradient descent to learn theta
       theta = gradientDescent(x, y, theta, alpha, num_iters) updates theta by
       taking num_iters gradient steps with learning rate alpha
    """
    theta=np.mat(theta).T
    X=np.mat(X)
    y=np.mat(y)
    print("zheli",X.shape,y.shape,theta.shape)
    # Initialize some useful values
    J_history = []
    m = y.size  # number of training examples

    for i in range(num_iters):
        #   ====================== YOUR CODE HERE ======================
        # Instructions: Perform a single gradient step on the parameter vector
        #               theta.
        #
        # Hint: While debugging, it can be useful to print out the values
        #       of the cost function (computeCost) and gradient here.
        #
        theta = theta - 1 / m * alpha * X.T * (X * theta - y)

        # ============================================================

        # Save the cost J in every iteration
        J_history.append(computeCost(X, y, theta.T))

    return theta, J_history
