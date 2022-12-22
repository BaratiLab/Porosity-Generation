# TEST ON GPU

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
im = torch.tensor(im, dtype=torch.float).unsqueeze(0).unsqueeze(0).cuda()


# Parameters for transforms

J = 3
L = 4
M, N = im.shape[-2], im.shape[-1]
j_max = 1
l_max = L
delta_k = 1

# kymatio scattering
from kymatio.phaseharmonics2d.phase_harmonics_k_bump \
    import PhaseHarmonics2d

wph_op = PhaseHarmonics2d(M, N, J, L, j_max, l_max, delta_k)
wph_op = wph_op.cuda()

Sim = wph_op(im)
#for key,val in Smeta.items():
#    print (key, "=>", val, ":", Sim[0,0,key,0,0,0], "+i ", Sim[0,0,key,0,0,1])
#print (Sim.shape)

# ---- Reconstruct marks. At initiation, every point has the average value of the marks.----#


#---- Optimisation with torch----#
# recontruct x by matching || Sx - Sx0 ||^2

x = torch.Tensor(1,1,N,N).normal_(std=0.1).cuda()
#x = torch.zeros(1,1,N,N).cuda()
#x[0,0,0,0]=2
x = Variable(x, requires_grad=True)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam([x], lr=.1)
nb_steps = 2000
for step in range(0, nb_steps + 1):
    optimizer.zero_grad()
    P = wph_op(x)
    loss = criterion(P, Sim)
    loss.backward()
    optimizer.step()
    if step % 100 == 0:
        print('step',step,'loss',loss)

# plot x
print(x.norm())

plt.imshow(x.detach().cpu().numpy().reshape((M,N)))
plt.show()

