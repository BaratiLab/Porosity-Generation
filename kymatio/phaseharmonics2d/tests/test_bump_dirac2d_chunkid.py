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

# kymatio scattering
from kymatio.phaseharmonics2d.phase_harmonics_k_bump_chunkid \
    import PhaseHarmonics2d

nb_chunks = 10
Sims = []
#n_coeff = 0
for chunk_id in range(nb_chunks+1):
    wph_op = PhaseHarmonics2d(M, N, J, L, delta_j, delta_l, delta_k, nb_chunks, chunk_id)
    Sim_ = wph_op(im) # (nb,nc,nb_channels,1,1,2)
    print('Sim_',Sim_.shape)
    #n_coeff += Sim_.shape[2]
    Sims.append(Sim_.numpy())# .squeeze())

Sim = np.concatenate(Sims,axis=2)
print(Sim.shape)
#Sim = Sim.view(1,1,-1,1,1,2)

#print(Sim.shape)

print('sum', Sim.sum())

plt.plot(Sim.squeeze()[...,1])
plt.show()
