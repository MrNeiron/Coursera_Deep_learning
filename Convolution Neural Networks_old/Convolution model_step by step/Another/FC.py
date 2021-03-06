import numpy as np

def Sigmoid(x):
    return (np.exp(x))/((np.exp(x)) + 1)
def SigmoidPrime(x):
    return Sigmoid(x) * (1 - Sigmoid(x))


def forwardPass(x, y, w, b, hyperparameters, activationFunction=Sigmoid):
    learningRate = hyperparameters["learningRate"]
    aS = []
    aS.append(x)
    zS = []
    zS.append(x)

    for i in range(len(w)):
        z = w[i].dot(aS[i]) + b[i]
        zS.append(z)
        a = activationFunction(z)
        aS.append(a)

    yHat = aS[-1]
    cache = [aS, y, w, b, hyperparameters]

    return yHat, cache


def backPropagation(caches2, primeActivationFunction=SigmoidPrime):
    (aS, y, w, b, hyperparameters) = caches2
    delta = (aS[-1] - y) * primeActivationFunction(aS[-1])
    nablaB = delta
    nablaW = delta.dot(aS[-2].T)

    for l in range(2, len(aS)):

        delta = w[-l + 1].T.dot(delta) * primeActivationFunction(aS[-l])
        nablaB = delta
        nablaW = delta.dot(aS[-l - 1].T)

    return nablaB, nablaW, delta