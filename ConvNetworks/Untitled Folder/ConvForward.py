import numpy as np
from Pad import PadZero

# GRADED FUNCTION: conv_forward

def FeedForwardConv(prevA,W,b,hyperparameters):
    """
    Implements the forward propagation for a convolution function
    
    Arguments:
    prevA -- output activations of the previous layer, numpy array of shape (m, nHprev, nWprev, nCprev)
    W -- Weights, numpy array of shape (f, f, nCprev, nC)
    b -- Biases, numpy array of shape (1, 1, 1, nC)
    hparameters -- python dictionary containing "stride" and "padding"
        
    Returns:
    Z -- conv output, numpy array of shape (m, nH, nW, nC)
    cache -- cache of values needed for the BackPropagationConv() function
    """
    
    ### START CODE HERE ###
    # Retrieve dimensions from A_prev's shape (≈1 line) 
    (m, oldNH, oldNW, nC) = prevA.shape
    
    # Retrieve dimensions from W's shape (≈1 line)
    (f,f,nCprev,nC) = W.shape
    
    # Retrieve information from "hparameters" (≈2 lines)
    stride = hyperparameters["stride"]
    pad = hyperparameters["padding"]
    
    # Compute the dimensions of the CONV output volume using the formula given above. Hint: use int() to floor. (≈2 lines)
    newNH = int((oldNH - f + 2*pad)/stride) + 1
    newNW = int((oldNH - f + 2*pad)/stride) + 1
    
    # Initialize the output volume Z with zeros. (≈1 line)
    Z = np.zeros((m,newNH, newNW, nC))
    #A = np.zeros((m,newNH, newNW, nC))
    
    if (pad != 0):
        # Create A_prev_pad by padding A_prev
        prevA = PadZero(prevA, pad)
    
    for m1 in range(m):                                                 # loop over the batch of training examples
        for i,h1 in enumerate(range(0,newNH,stride)):                   # loop over vertical axis of the output volume# Select ith training example's padded activation
            for j,w1 in enumerate(range(0,newNW, stride)):              # loop over horizontal axis of the output volume
                for c1 in range(nC):                                     # loop over channels (= #filters) of the output volume
                    Z[m1,i,j,c1] = np.sum((prevA[m1,h1:h1+f,w1:w1+f,:] * W[...,c1]) + b[...,c1])#[...,с1] нужно, чтобы размерности совпали
                    #A[m1,i,j,c1] = activationFunction(Z[m1,i,j,c1])

    ### END CODE HERE ###
    
    # Making sure your output shape is correct
    assert ((m,newNH,newNW,nC) == Z.shape)
    
    # Save information in "cache" for the backprop
    cache = [Z, W, b, hyperparameters]
    
    return Z, cache