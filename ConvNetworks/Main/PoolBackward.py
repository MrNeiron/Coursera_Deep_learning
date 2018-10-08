import numpy as np
from CreateMaskForWindow import CreateMaskForWindow
from DistributeValue import DistributeValue

def PoolBackward(dA, cache, mode = None):

    (Aprev, hyperparameters, modeCache) = cache
    
    mode = "max" if mode == None and modeCache == None else modeCache if mode == None else mode 
    
    stride = hyperparameters["stride"]
    f = hyperparameters["f"]
    
    (m, nHprev, nWprev, nCprev) = Aprev.shape
    (m, nH, nW, nC) = dA.shape
    
    dAprev = np.zeros(Aprev.shape)
    
    for m1 in range(m):                                               
        for i,h1 in enumerate(range(0,nHprev-f+1, stride)):           
            for j,w1 in enumerate(range(0,nWprev-f+1, stride)):       
                for c1 in range(nC):                                  


                    if mode == "max": 
                        dAprev[m1,h1:h1+f,w1:w1+f,c1] += CreateMaskForWindow(Aprev[m1,h1:h1+f,w1:w1+f,c1]) * dA[m1,i,j,c1]
                    elif mode == "average":
                        s = DistributeValue(dA[m1,i,j,c1], (f,f))
                        dAprev[m1,h1:h1+f,w1:w1+f,c1] += s

    assert(dAprev.shape == Aprev.shape)
    
    return dAprev

