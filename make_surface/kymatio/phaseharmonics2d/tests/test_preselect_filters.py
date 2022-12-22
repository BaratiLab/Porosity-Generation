
import scipy.optimize as opt
import scipy.io as sio

import torch
from torch.autograd import Variable, grad

from time import time
import gc

size=256

# --- Dirac example---#
data = sio.loadmat('./data/demo_toy7d_N' + str(size) + '.mat')
im = data['imgs']
im = torch.tensor(im, dtype=torch.float).unsqueeze(0).unsqueeze(0)

# Parameters for transforms
J = 8
L = 8
M, N = im.shape[-2], im.shape[-1]
delta_j = 1
delta_l = L/2
delta_k = 1
delta_n = 1
nb_chunks = 20
nb_restarts = 10
nGPU = 1
factr = 1e5

#from kymatio.phaseharmonics2d.phase_harmonics_k_bump_chunkid_pershift \
#    import PHkPerShift2d
from kymatio.phaseharmonics2d.phase_harmonics_k_bump_chunkid_scaleinter \
    import PhkScaleInter2d

dn1 = 0
dn2 = 0
chunk_id = 1
devid = 0
#wph_op = PHkPerShift2d(M, N, J, L, dn1, dn2, delta_l, J, chunk_id, devid)
wph_op = PhkScaleInter2d(M, N, J, L, delta_j, delta_l, delta_k, nb_chunks, chunk_id, devid)
Sim = wph_op(im)

print('shape',Sim.shape)
print('value',Sim.squeeze())
