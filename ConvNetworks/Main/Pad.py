import numpy as np

def PadZero(a, pad):
    return np.pad(a,((0,0),(pad,pad),(pad,pad),(0,0)), "constant", constant_values=0)

