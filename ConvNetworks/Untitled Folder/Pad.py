import numpy as np
#import matplotlib.pyplot as plt

def PadZero(a, pad):
    return np.pad(a,((0,0),(pad,pad),(pad,pad),(0,0)), "constant", constant_values=0)

'''
x = np.random.randint(1,3,(1,8,8,1))
print("x{}: \n{}".format(x.shape,x[0,:,:,0]))
z = PadZero(x, 2)
print("x{}: \n{}".format(z.shape,z[0,:,:,0]))


fig, ax = plt.subplots(1,2)
ax[0].set_title("x{}".format(x.shape))
ax[0].imshow(x[0,:,:,0])
ax[1].set_title("z{}".format(z.shape))
ax[1].imshow(x[0,:,:,0])
'''
