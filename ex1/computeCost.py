import numpy as np

def computeCost(X, y, theta):
    """
       computes the cost of using theta as the parameter for linear 
       regression to fit the data points in X and y
    """
    theta=np.mat(theta).T
    X=np.mat(X)
    y=np.mat(y)


    m = y.size
    print(X.shape,y.shape,theta.shape)
# ====================== YOUR CODE HERE ======================
# Instructions: Compute the cost of a particular choice of theta
#               You should set J to the cost.
    J = 1 / (2 * m) * (X * theta - y).T * (X * theta - y)
    print(J.shape)
# =========================================================================

    return J


