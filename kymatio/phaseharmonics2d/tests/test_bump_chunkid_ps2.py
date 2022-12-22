# TEST ON CPU

#import pandas as pd
import numpy as np

import scipy.optimize as opt
import scipy.io as sio

import torch
from torch.autograd import Variable, grad

from time import time

size=256

# --- Dirac example---#
data = sio.loadmat('./data/demo_toy7d_N' + str(size) + '.mat')
im = data['imgs']
im = torch.tensor(im, dtype=torch.float).unsqueeze(0).unsqueeze(0)

J = 8
L = 8
M, N = im.shape[-2], im.shape[-1]
delta_j = 3 # int(sys.argv[2])
delta_l = L/2
delta_k = 1
delta_n = 0 # int(sys.argv[3])
nb_chunks = 40 # int(sys.argv[4])

from kymatio.phaseharmonics2d.phase_harmonics_k_bump_chunkid_pershift \
    import PHkPerShift2d
from kymatio.phaseharmonics2d.phase_harmonics_k_bump_chunkid_scaleinter \
    import PhkScaleInter2d

def print_Sim(Sim,wph_idx,L2):
    nbc = Sim.shape[2]
    idx_list = []
    val_list = []
    for idxbc in range(nbc):
        j1 = wph_idx['la1'][idxbc]//L2
        theta1 = wph_idx['la1'][idxbc]%L2
        k1 = wph_idx['k1'][0,idxbc,0,0]
        j2 = wph_idx['la2'][idxbc]//L2
        theta2 = wph_idx['la2'][idxbc]%L2
        k2 = wph_idx['k2'][0,idxbc,0,0]
        val = (int(j1),int(theta1),int(k1),int(j2),int(theta2),int(k2))
        #print(idxbc, "=>" , val,  float(Sim[0,0,idxbc,0,0,0]),  "+i ",float(Sim[0,0,idxbc,0,0,1]) )
        idx_list.append(val)
        val_list.append(float(Sim[0,0,idxbc,0,0,0]) + 1j*float(Sim[0,0,idxbc,0,0,1]))
    return idx_list, val_list

nCov = 0
devid = 0
L2=L*2

for dn1 in range(-delta_n,delta_n+1):
    for dn2 in range(-delta_n,delta_n+1):
        if dn1**2+dn2**2 <= delta_n**2:
            for chunk_id in range(J):
                if dn1==0 and dn2==0:
                    wph_op = PHkPerShift2d(M, N, J, L, dn1, dn2, delta_l, J, chunk_id)
                else:
                    wph_op = PHkPerShift2d(M, N, J, L, dn1, dn2, 0, J, chunk_id)
                im_ = im
                Sim_ = wph_op(im_) # (nb,nc,nb_channels,1,1,2)
                idx_list, val_list = print_Sim(Sim_,wph_op.this_wph,L2)
                print('save dn1 dn2', dn1, dn2)
                sio.savemat('/home/zsx/cosmo_wph/tests/test_bump_chunk_id_ps2_chunk_id' +
                            str(chunk_id) + '_dn' + str(dn1) + str(dn2) + '.mat',
                            {'idx_list':idx_list, 'val_list':val_list})

'''
for chunk_id in range(nb_chunks):
    wph_op = PhkScaleInter2d(M, N, J, L, delta_j, delta_l, delta_k, nb_chunks, chunk_id, devid)
    im_ = im
    Sim_ = wph_op(im_) # (nb,nc,nb_channels,1,1,2)
    idx_list, val_list = print_Sim(Sim_,wph_op.this_wph,L2)
    print('save chunk id', chunk_id)
    sio.savemat('/home/zsx/cosmo_wph/tests/test_bump_chunk_id_ps2_chunkid' + str(chunk_id) + '.mat', {'idx_list':idx_list, 'val_list':val_list})

wph_op = PhkScaleInter2d(M, N, J, L, delta_j, delta_l, delta_k, nb_chunks, nb_chunks, devid)
SimJ = wph_op(im)
print(SimJ.squeeze()[0])
'''
