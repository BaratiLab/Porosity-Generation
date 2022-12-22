# TEST ON CPU

#import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import scipy.optimize as opt

import torch
from torch.autograd import Variable, grad

from time import time

#---- create image without/with marks----#

size=32

# --- Dirac example---#

im = np.zeros((size,size))
im[15,15] = 1
im = torch.tensor(im, dtype=torch.float).unsqueeze(0).unsqueeze(0)


# Parameters for transforms

J = 3
L = 4
M, N = im.shape[-2], im.shape[-1]
delta_j = 1
delta_l = L
delta_k = 1
oversampling = 1

# kymatio scattering
from kymatio.phaseharmonics2d.phase_harmonics_k_bump_sub \
    import PhaseHarmonics2d


wph_op = PhaseHarmonics2d(M, N, J, L, delta_j, delta_l, delta_k, oversampling)

L2 = L*2
Sim = wph_op(im) # (nb,nc,nb_channels,1,1,2)
print('SUM',Sim.sum())


# last channel is l2
#print(nbc-1, "=> phiJ " ,  float(Sim[0,0,nbc-1,0,0,0]),  "+i ",float(Sim[0,0,nbc-1,0,0,1]) )

#plt.plot(Sim.squeeze().numpy()[...,0])
#plt.show()
