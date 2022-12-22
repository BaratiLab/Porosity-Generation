# TEST ON CPU

#import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt

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
j_max = 1
l_max = L

# kymatio scattering
from kymatio.phaseharmonics2d.phase_harmonics_k_idx \
    import PhaseHarmonics2d

wph_op = PhaseHarmonics2d(M, N, J, L, j_max, l_max)

Sim = wph_op(im) # (nb,nc,nb_channels,1,1,2)
nbc = Sim.shape[2]
for idxbc in range(nbc):
    j1 = wph_op.idx_wph['la1'][idxbc]//L
    theta1 = wph_op.idx_wph['la1'][idxbc]%L
    k1 = wph_op.idx_wph['k1'][0,idxbc,0,0]
    j2 = wph_op.idx_wph['la2'][idxbc]//L
    theta2 = wph_op.idx_wph['la2'][idxbc]%L
    k2 = wph_op.idx_wph['k2'][0,idxbc,0,0]
    val = (j1,theta1,k1,j2,theta2,k2)
    print(idxbc, "=>" , val,  Sim[0,0,idxbc,0,0,0],  "+i ",Sim[0,0,idxbc,0,0,1] )
    
