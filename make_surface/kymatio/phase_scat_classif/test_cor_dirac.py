import torch
import torchvision
from utils import rgb2yuv

from representation_complex_inv import compute_phase_harmonic_cor_inv

from complex_utils import complex_log
from utils import mean_std, standardize_feature

import numpy as np

# dirac

size=32

# --- Dirac example---#

im = np.zeros((size,size))
im[15,15] = 1
im = torch.tensor(im, dtype=torch.float).unsqueeze(0).unsqueeze(0) # (1,1,size,size)

J = 3
L = 4
batch_size = 1
delta = 1
l_max = L

Sim = compute_phase_harmonic_cor_inv(im, J, L, delta, l_max, batch_size)
Sim_sz = Sim.shape
print('Sim sz:',Sim_sz)
for idxc in range(Sim_sz[2]):
    print('Sim:',idxc,Sim[0,0,idxc,0,0,0],'+i',Sim[0,0,idxc,0,0,1])
    
