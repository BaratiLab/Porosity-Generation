
import numpy as np

import scipy.optimize as opt
import scipy.io as sio

import torch
from torch.autograd import Variable, grad

from time import time

size=32

im = torch.randn(1,1,size,size) # ,dtype=torch.float)

from kymatio.phaseharmonics2d.wavelet_covariance_pershift2d \
    import WaveletCovPerShift2d
from kymatio.phaseharmonics2d.wavelet_covariance_fftshift2d \
    import WaveletCovFFTShift2d

M=size
N=size
J=3
L=4
delta_n = 4
dn_mode = 0
nb_chunks = J

im_ = im
for chunk_id in range(J+1):
    wph_op = WaveletCovPerShift2d(M, N, J, L, delta_n, dn_mode, J, chunk_id)
    #wph_op = wph_op.cuda()
    Sim_ = wph_op(im_) # (nb,nc,nb_channels,1,1,2)
    wph_op2 = WaveletCovFFTShift2d(M,N,J,L,delta_n,dn_mode,J,chunk_id)
    Sim2_ = wph_op2(im_)
    
    diff = Sim_-Sim2_
    print(chunk_id,diff.abs().max())
    
        
