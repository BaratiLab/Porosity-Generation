
import numpy as np
import scipy.io as sio
import torch

size = 256

data = sio.loadmat('./data/demo_toy7d_N' + str(size) + '.mat')
im = data['imgs']
im = torch.tensor(im, dtype=torch.float).unsqueeze(0).unsqueeze(0)

N=size
M=N
shift1 = 0
shift2 = 0
from kymatio.phaseharmonics2d.backend import PeriodicShift2D, Pad
pershift = PeriodicShift2D(M,N,shift1,shift2)
pad = Pad(0, pre_pad=False)

im_ = pershift(pad(im))
print(im_.shape)

import matplotlib.pyplot as plt

plt.subplot(121)
plt.imshow(im.view(N,N))
plt.subplot(122)
plt.imshow(im_[...,0].view(N,N))
plt.show()
