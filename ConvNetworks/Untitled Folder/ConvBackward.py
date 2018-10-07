import numpy as np
from Pad import PadZero

def BackPropagationConv(cache, dZ):
    """
    Implement the backward propagation for a convolution function
    
    Arguments:
    dZ -- gradient of the cost with respect to the output of the conv layer (Z), numpy array of shape (m, nH, nW, nC)
    cache -- cache of values needed for the BackPropagationConv(), output of BackPropagationConv()
    
    Returns:
    dA_prev -- gradient of the cost with respect to the input of the conv layer (Aprev),
               numpy array of shape (m, nH_prev, nWprev, nCprev)
    dW -- gradient of the cost with respect to the weights of the conv layer (W)
          numpy array of shape (f, f, nCprev, nC)
    db -- gradient of the cost with respect to the biases of the conv layer (b)
          numpy array of shape (1, 1, 1, nC)
    """
    
    ### START CODE HERE ###
    # Retrieve information from "cache"
    (Aprev,W,b,hyperparameters) = cache
    
    # Retrieve dimensions from Aprev's shape
    (m,nHprev,nWprev,nCprev) = Aprev.shape
    
    # Retrieve dimensions from W's shape
    (f,f,nCprev,nC) = W.shape
    
    # Retrieve information from "hyperparameters"
    stride = hyperparameters["stride"]
    pad = hyperparameters["padding"]
    
    # Retrieve dimensions from dZ's shape
    (m, nH, nW, nC) = dZ.shape
    
    # Initialize dAprev, dW, db with the correct shapes
    dAprev = np.zeros(Aprev.shape)
    dW = np.zeros(W.shape)
    dB = np.zeros((1,1,1,nC))
    
    # Pad Aprev and dAprev
    AprevPad = PadZero(Aprev, pad)
    dAprevPad = PadZero(dAprev, pad)
    
    for m1 in range(m):                                                 # loop over the training examples
        for i,h1 in enumerate(range(0,nH-f+1,stride)):                  # loop over vertical axis of the output volume
            for j,w1 in enumerate(range(0,nW-f+1,stride)):              # loop over horizontal axis of the output volume
                for c1 in range(nC):                                    # loop over the channels of the output volume
                    
                    print("\n\nm1: {}\ni:{} h1:{}\nj:{} w1:{}\nc1:{}".format(m1,i,h1,j,w1,c1))
                    print("W1: \n",dW[...,c1])
                    print("a: \n", dAprevPad[m1,h1:h1+f,w1:w1+f,c1])
                    print("delta: \n",dZ[m1,i,j,c1])
                    
                    #dAprevPad[m1,h1:h1+f,w1:w1+f,:] += W[...,c1] * dZ[m1, h1:h1+f, w1:w1+f, c1]
                    dAprevPad[m1,h1:h1+f,w1:w1+f,:] += W[...,c1] * dZ[m1, i, j, c1]
                    #da_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] += W[:,:,:,c] * dZ[i, h, w, c]
                    dW[...,c1] += AprevPad[m1,h1:h1+f,w1:w1+f,:] * dZ[m1,i,j,c1]
                    dB[...,c1] += dZ[m1,i,j,c1]
                    print("W: \n",dW[...,c1])
    
    dAprev = dAprevPad[:,pad:-pad,pad:-pad,:]
    print("dAprev{}: \n{}".format(dAprev.shape, dAprev))
    # Making sure your output shape is correct
    assert (dAprev.shape == (m, nHprev,nWprev,nCprev))       
                    
    return dAprev, dW, dB