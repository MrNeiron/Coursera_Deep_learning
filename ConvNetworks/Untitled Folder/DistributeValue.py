import numpy as np

def DistributeValue(dz, shape):
    (n_H, n_W) = shape
    
    average = dz / (n_H * n_W)
    
    a = np.ones(shape) * average
    
    return a

