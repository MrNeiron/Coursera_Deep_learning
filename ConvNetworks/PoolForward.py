import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)

def MaxPool(matrix):
    return matrix.max()
def AveragePool(matrix):
    return int(matrix.mean())


def PoolForward(a, f, parameter, hyperparameters):
    stride = hyperparameters["stride"]

    (m, nH, nW, nC) = a.shape  # Input shape of the tensor

    newNH = int((nH - f) / stride) + 1
    newNW = int((nH - f) / stride) + 1

    p = np.zeros((m, newNH, newNW, nC), dtype=np.int32)
    for m1 in range(m):
        for i, h1 in enumerate(range(0, nH, stride)):
            for j, w1 in enumerate(range(0, nW, stride)):
                for c1 in range(nC):
                    try:
                        p[m1, i, j, c1] = MaxPool(
                            a[m1, h1:h1 + f, w1:w1 + f, c1]) if parameter == "max" else AveragePool(
                            a[m1, h1:h1 + f, w1:w1 + f, c1])
                    except:
                        break
    return p


m = 1
nH = 8
nW = 8
nC = 1
poolSize = 2
stride = 2
hyperparameters ={"stride": stride}
x = np.random.randint(0,3,(m,nH,nW,nC))


fig, ax1 = plt.subplots(1,1)
ax1.set_title("x{}".format(x.shape))
ax1.imshow(x[0,:,:,0])

p = PoolForward(x, poolSize, "max",hyperparameters)

print("x:\n",x[0,:,:,0])
print("p:\n",p[0,:,:,0])
fig, ax1 = plt.subplots(1,1)
ax1.set_title("p{}".format(p.shape))
ax1.imshow(p[0,:,:,0])