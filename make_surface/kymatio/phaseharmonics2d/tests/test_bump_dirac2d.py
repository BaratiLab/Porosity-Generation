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
from kymatio.phaseharmonics2d.phase_harmonics_k_bump \
    import PhaseHarmonics2d

wph_op = PhaseHarmonics2d(M, N, J, L, delta_j, delta_l, delta_k)

L2 = L*2
Sim = wph_op(im) # (nb,nc,nb_channels,1,1,2)
print('sum', Sim.sum())

#nbc = Sim.shape[2]
#for idxbc in range(nbc-1):
#    j1 = wph_op.idx_wph['la1'][idxbc]//L2
#    theta1 = wph_op.idx_wph['la1'][idxbc]%L2
#    k1 = wph_op.idx_wph['k1'][0,idxbc,0,0]
#    j2 = wph_op.idx_wph['la2'][idxbc]//L2
#    theta2 = wph_op.idx_wph['la2'][idxbc]%L2
#    k2 = wph_op.idx_wph['k2'][0,idxbc,0,0]
#    val = (int(j1),int(theta1),int(k1),int(j2),int(theta2),int(k2))
#    if idxbc == 384: 
#        print(idxbc, "=>" , val,  float(Sim[0,0,idxbc,0,0,0]),  "+i ",float(Sim[0,0,idxbc,0,0,1]) )

# last channel is l2
#print(nbc-1, "=> phiJ " ,  float(Sim[0,0,nbc-1,0,0,0]),  "+i ",float(Sim[0,0,nbc-1,0,0,1]) )

#plt.plot(Sim.squeeze().numpy()[...,0])
#plt.show()
