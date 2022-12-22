# TEST ON GPU
import os,sys
import numpy as np
import scipy.optimize as opt
import torch
from torch.autograd import Variable, grad

size = 32
im = np.random.randn(size,size)*10
im = torch.tensor(im, dtype=torch.float).unsqueeze(0).unsqueeze(0).cuda()

# Parameters for transforms
J = 5
L = 4
M, N = im.shape[-2], im.shape[-1]
delta_j = 1
delta_l = int(L/2)
delta_k = 1
nb_chunks = 1
submean = 1
stdnorm = 1

from kymatio.phaseharmonics2d.phase_harmonics_k_bump_non_isotropic_bigdj import PhaseHarmonics2d

for chunk_id in range(nb_chunks):
    wph_op = PhaseHarmonics2d(M, N, J, L, delta_j, delta_l, delta_k, nb_chunks, chunk_id, submean=submean, stdnorm=stdnorm)
    wph_op = wph_op.cuda()
    Sim_ = wph_op(im)
    print('Sim max,min is',Sim_.max(),Sim_.min())
    print('nb of wph coefficients',Sim_.shape[2])
