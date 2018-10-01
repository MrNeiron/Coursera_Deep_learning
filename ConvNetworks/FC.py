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

    cache = (aS, y, w, b, hyperparameters)
    yHat = aS[-1]
    return yHat, cache


def backPropagation(caches2, primeActivationFunction=SigmoidPrime):
    aS = caches2[0][0]
    y = caches2[0][1]
    w = caches2[0][2]
    b = caches2[0][3]
    delta = (aS[-1] - y) * primeActivationFunction(aS[-1])
    nablaB = delta
    nablaW = delta.dot(aS[-2].T)

    for l in range(2, len(aS)):
        '''
        print("w[{}]{}: \n{}".format(-l+1,w[-l+1].shape, w[-l+1]))
        print("delta1{}: \n{}".format(delta.shape, delta))
        print("aS[{}]{}: \n{}".format(-l, aS[-l].shape, aS[-l]))
        '''
        delta = w[-l + 1].T.dot(delta) * primeActivationFunction(aS[-l])
        nablaB = delta
        nablaW = delta.dot(aS[-l - 1].T)
        # print("delta2{}: \n{}".format(delta.shape, delta))

    return nablaB, nablaW