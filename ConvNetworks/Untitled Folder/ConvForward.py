import numpy as np
from Pad import PadZero

def ConvForward(prevA,W,b,hyperparameters):

    (m, oldNH, oldNW, nC) = prevA.shape

    (f,f,nCprev,nC) = W.shape

    stride = hyperparameters["stride"]
    pad = hyperparameters["padding"]

    newNH = int((oldNH - f + 2*pad)/stride) + 1
    newNW = int((oldNH - f + 2*pad)/stride) + 1

    Z = np.zeros((m,newNH, newNW, nC))
    
    if (pad != 0):
        prevA = PadZero(prevA, pad)
    
    for m1 in range(m):                                                
        for i,h1 in enumerate(range(0,newNH,stride)):                   
            for j,w1 in enumerate(range(0,newNW, stride)):             
                for c1 in range(nC):                                     
                    Z[m1,i,j,c1] = np.sum((prevA[m1,h1:h1+f,w1:w1+f,:] * W[...,c1]) + b[...,c1])#[...,с1] нужно, чтобы размерности совпали

    assert ((m,newNH,newNW,nC) == Z.shape)

    cache = [Z, W, b, hyperparameters]
    
    return Z, cache