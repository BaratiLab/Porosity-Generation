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
delta_j = 1
delta_l = L
delta_k = 1

# kymatio scattering
from kymatio.phaseharmonics2d.phase_harmonics_k_bump_chunkid \
    import PhaseHarmonics2d


nb_chunks = 10
Sims = []
factr = 1e3
wph_ops = []
for chunk_id in range(nb_chunks+1):
    wph_op = PhaseHarmonics2d(M, N, J, L, delta_j, delta_l, delta_k, nb_chunks, chunk_id)
    wph_op = wph_op.cuda()
    wph_ops.append(wph_op)
    Sim_ = wph_op(im)*factr # (nb,nc,nb_channels,1,1,2)
    Sims.append(Sim_)
    
# ---- Reconstruct marks. At initiation, every point has the average value of the marks.----#
#---- Trying scipy L-BFGS ----#
def obj_fun(x,chunk_id):
    if x.grad is not None:
        x.grad.data.zero_()
    wph_op = wph_ops[chunk_id]
    p = wph_op(x)*factr
    diff = p-Sims[chunk_id]
    loss = torch.mul(diff,diff).mean()
    return loss

grad_err = im.clone()

def grad_obj_fun(x):
    loss = 0
    global grad_err
    grad_err[:] = 0
    for chunk_id in range(nb_chunks+1):
        loss = loss + obj_fun(x,chunk_id)
        grad_err_, = grad([loss],[x], retain_graph=True)
        grad_err = grad_err + grad_err_
    return loss, grad_err

count = 0
def fun_and_grad_conv(x):
    x_t = torch.reshape(torch.tensor(x, requires_grad=True,dtype=torch.float),
                        (1,1,size,size)).cuda()
    loss, grad_err = grad_obj_fun(x_t)
    global count
    count += 1
    if count%40 == 1:
        print(loss)
    return  loss.cpu().item(), np.asarray(grad_err.reshape(size**2).cpu().numpy(), dtype=np.float64)

#float(loss)

x = torch.Tensor(1, 1, N, N).normal_(std=0.1)
#x[0,0,0,0] = 2
#x = x.clone().detach().requires_grad_(True) # torch.tensor(x, requires_grad=True)
x0 = x.reshape(size**2).detach().numpy()
x0 = np.asarray(x0, dtype=np.float64)

res = opt.minimize(fun_and_grad_conv, x0, method='L-BFGS-B', jac=True, tol=None,
                   callback=callback_print,
                   options={'maxiter': 500, 'gtol': 1e-14, 'ftol': 1e-14, 'maxcor': 100})
final_loss, x_opt, niter, msg = res['fun'], res['x'], res['nit'], res['message']

im_opt = np.reshape(x_opt, (size,size))
#tensor_opt = torch.tensor(im_opt, dtype=torch.float).unsqueeze(0).unsqueeze(0)
plt.figure()
#im_opt = np.reshape(x_opt, (size,size))
plt.imshow(im_opt)
plt.show()
