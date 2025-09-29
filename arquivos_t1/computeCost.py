import numpy as np


def compute_cost(X, y, theta):
    
    m = y.size  

    # Produto escalar para predição inicial
    pred = X.dot(theta)

    sq_e = (pred - y) ** 2
    cost = (1 / (2 * m)) * np.sum(sq_e)

    return cost
