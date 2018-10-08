import numpy as np
from Pad import PadZero

def ConvBackward(cache, dZ):

    (Aprev,W,b,hyperparameters) = cache

    (m,nHprev,nWprev,nCprev) = Aprev.shape

    (f,f,nCprev,nC) = W.shape

    stride = hyperparameters["stride"]
    pad = hyperparameters["padding"]

    (m, nH, nW, nC) = dZ.shape

    dAprev = np.zeros((m,nHprev,nWprev,nCprev))
    dW = np.zeros((f,f,nCprev,nC))
    dB = np.zeros((1,1,1,nC))

    AprevPad = PadZero(Aprev, pad)

    dAprevPad = PadZero(dAprev, pad)
    
    for m1 in range(m):                                           
        for i,h1 in enumerate(range(0,nH,stride)):                  
            for j,w1 in enumerate(range(0,nW,stride)):              
                for c1 in range(nC):                                    

                    dAprevPad[m1,h1:h1+f,w1:w1+f,:] += W[:,:,:,c1] * dZ[m1, i, j, c1]
                    
                    dW[...,c1] += AprevPad[m1,h1:h1+f,w1:w1+f,:] * dZ[m1,i,j,c1]
                    dB[...,c1] += dZ[m1,i,j,c1]
    
    dAprev = dAprevPad[:,pad:-pad,pad:-pad,:]

    assert (dAprev.shape == (m,nHprev,nWprev,nCprev))       
                    
    return dAprev, dW, dB