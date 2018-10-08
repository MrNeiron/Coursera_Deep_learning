import numpy as np

def CreateMaskForWindow(x):

    mask = x == np.max(x)
    
    return mask

