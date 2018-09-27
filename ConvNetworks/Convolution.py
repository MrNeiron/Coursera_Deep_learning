import numpy as np

def SingleSlice(tensor, weights, bias):
    return np.sum((tensor * weights) + bias)