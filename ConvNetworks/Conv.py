import numpy as np
from Pad import PadZero

def Sigmoid(x):
    return (np.exp(x))/((np.exp(x)) + 1)
def SigmoidPrime(x):
    return Sigmoid(x) * (1 - Sigmoid(x))

def Empty(x):
    return x


def feedForwardConv(prevA, w, b, hyperparameters, activationFunction=Sigmoid):
    (m, oldNH, oldNW, nC) = prevA.shape
    (f, f, nC) = w.shape
    stride = hyperparameters["stride"]
    pad = hyperparameters["padding"]

    newNH = int((oldNH - f + 2 * pad) / stride) + 1
    newNW = int((oldNH - f + 2 * pad) / stride) + 1

    z = np.zeros((m, newNH, newNW, nC))
    a = np.zeros((m, newNH, newNW, nC))

    if (pad != 0):
        prevA = PadZero(prevA, pad)

    for m1 in range(m):
        for i, h1 in enumerate(range(0, newNH, stride)):
            for j, w1 in enumerate(range(0, newNW, stride)):
                for c in range(nC):
                    z[m1, i, j, c] = np.sum((prevA[m1, h1:h1 + f, w1:w1 + f, c] * w[
                        ..., c]) + b)  # [...,с] нужно, чтобы размерности совпали
                    a[m1, i, j, c] = activationFunction(z[m1, i, j, c])

    assert (m, newNH, newNW, nC) == a.shape

    cache = [a, w, b, hyperparameters]

    return a, cache


def backPropagationConv(cache, delta, activationFunctionPrime=SigmoidPrime):
    (a, w, b, hyperparameters) = cache
    (m, nH, nW, nC) = a.shape
    (f, f, nC) = w.shape
    stride = hyperparameters["stride"]
    pad = hyperparameters["padding"]
    nablaW = np.zeros(w.shape)

    for m1 in range(m):
        for i, h1 in enumerate(range(0, nH - f + 1, stride)):
            for j, w1 in enumerate(range(0, nW - f + 1, stride)):
                for c1 in range(nC):
                    print("\n\nm1: {}\ni:{} h1:{}\nj:{} w1:{}\nc1:{}".format(m1, i, h1, j, w1, c1))
                    print("W1: \n", nablaW[:, :, c1])
                    print("a: \n", a[m1, h1:h1 + f, w1:w1 + f, c1])
                    print("delta: \n", delta[h1:h1 + f, w1:w1 + f])

                    newW = a[m1, h1:h1 + f, w1:w1 + f, c1] * delta[h1:h1 + f, w1:w1 + f]
                    nablaW[:, :, c1] += newW

                    print("W2: \n", newW)
                    print("W: \n", nablaW[:, :, c1])

    return nablaW