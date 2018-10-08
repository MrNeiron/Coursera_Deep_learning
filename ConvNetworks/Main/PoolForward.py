import numpy as np

def PoolForward(Aprev, hyperparameters, mode = "max"):

    stride = hyperparameters["stride"]
    f= hyperparameters['f']

    (m, nH, nW, nC) = Aprev.shape
    
    newNH = int((nH - f)/stride)+1
    newNW = int((nH - f)/stride)+1

    A = np.zeros((m,newNH, newNW, nC), dtype=np.int32)

    for m1 in range(m):                                            
        for i,h1 in enumerate(range(0,nH,stride)):                  
            for j,w1 in enumerate(range(0,nW,stride)):              
                for c1 in range(nC):                                
                    try:
                        if mode == "max":
                            A[m1,i,j,c1] = np.max(Aprev[m1,h1:h1+f,w1:w1+f,c1]) 
                        elif mode == "average":
                            A[m1,i,j,c1] = int(np.mean(Aprev[m1,h1:h1+f,w1:w1+f,c1]))
                    except:
                        break

    cache = (Aprev, hyperparameters, mode)
    
    return A, cache