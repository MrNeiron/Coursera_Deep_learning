import numpy as np
#import matplotlib.pyplot as plt

#np.random.seed(1)


def PoolForward(a, hyperparameters, mode="max"):
    stride = hyperparameters["stride"]
    f = hyperparameters['f']

    (m, nH, nW, nC) = a.shape  # Input shape of the tensor

    newNH = int((nH - f) / stride) + 1  # Output height of the tensor
    newNW = int((nH - f) / stride) + 1  # Output width of the tensor

    p = np.zeros((m, newNH, newNW, nC), dtype=np.int32)  # Output tensor

    for m1 in range(m):
        for i, h1 in enumerate(range(0, nH, stride)):
            for j, w1 in enumerate(range(0, nW, stride)):
                for c1 in range(nC):
                    try:
                        if mode == "max":
                            p[m1, i, j, c1] = np.max(a[m1, h1:h1 + f, w1:w1 + f, c1])
                        elif mode == "average":
                            p[m1, i, j, c1] = int(np.mean(a[m1, h1:h1 + f, w1:w1 + f, c1]))
                    except:
                        break
    cache = (a, hyperparameters, mode)  # Parameters for back propagation

    return p, cache

'''
m = 1
nH = 8
nW = 8
nC = 1
poolSize = 2
stride = 2
hyperparameters ={"stride": stride, 'f':poolSize}
x = np.random.randint(0,3,(m,nH,nW,nC))

fig, ax1 = plt.subplots(1,1)
ax1.set_title("x{}".format(x.shape))
ax1.imshow(x[0,:,:,0])

p,cache = PoolForward(x, hyperparameters)

print("x:\n",x[0,:,:,0])
print("p:\n",p[0,:,:,0])
fig, ax1 = plt.subplots(1,1)
ax1.set_title("p{}".format(p.shape))
ax1.imshow(p[0,:,:,0])
'''