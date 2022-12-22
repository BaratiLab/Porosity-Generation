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
max_chunk = 100

# kymatio scattering
from kymatio.phaseharmonics2d.phase_harmonics_k_bump_chunks \
    import PhaseHarmonics2d

wph_op = PhaseHarmonics2d(M, N, J, L, j_max, l_max, delta_k, max_chunk)
wph_op = wph_op.cuda()

factr = 1e3
Sim = wph_op(im)*factr
#for key,val in Smeta.items():
#    print (key, "=>", val, ":", Sim[0,0,key,0,0,0], "+i ", Sim[0,0,key,0,0,1])
#print (Sim.shape)

# ---- Reconstruct marks. At initiation, every point has the average value of the marks.----#


#---- Trying scipy L-BFGS ----#

def f(a, b):
    p = wph_op(a)*factr
    diff = p-b
    return (torch.mul(diff,diff).mean(), p)

def closure():
    loss, Sx = f(x, Sim)
    loss_vals.append(loss.item())
    loss.backward()
    return loss


def obj_fun(x):
    if x.grad is not None:
        x.grad.data.zero_()
    loss, Sx = f(x, Sim)
    return loss


def grad_obj_fun(x):
    loss = obj_fun(x)
    grad_err, = grad([loss],[x], retain_graph=True)
    return loss, grad_err

def fun_and_grad_conv(x):
    x_t = torch.reshape(torch.tensor(x, requires_grad=True,dtype=torch.float),
                        (1,1,size,size)).cuda()
    loss, grad_err = grad_obj_fun(x_t)
    return  loss.cpu().item(), np.asarray(grad_err.reshape(size**2).cpu().numpy(), dtype=np.float64)

count = 0

def callback_print(x):
    global count
    count +=1
    if count%40 == 1:
        x_t = torch.reshape(torch.tensor(x, requires_grad=True,dtype=torch.float),
                            (1,1,size,size)).cuda()
        p = wph_op(x_t)*factr
        diff = p-Sim
        loss = torch.mul(diff,diff).mean()
        print(loss)
        return loss

x = torch.Tensor(1, 1, N, N).normal_(std=0.1)
#x[0,0,0,0] = 2
x = x.clone().detach().requires_grad_(True) # torch.tensor(x, requires_grad=True)
x0 = x.reshape(size**2).detach().numpy()
x0 = np.asarray(x0, dtype=np.float64)


res = opt.minimize(fun_and_grad_conv, x0, method='L-BFGS-B', jac=True, tol=None,
                   callback=callback_print,
                   options={'maxiter': 500, 'gtol': 1e-14, 'ftol': 1e-14, 'maxcor': 100})
final_loss, x_opt, niter, msg = res['fun'], res['x'], res['nit'], res['message']

im_opt = np.reshape(x_opt, (size,size))
tensor_opt = torch.tensor(im_opt, dtype=torch.float).unsqueeze(0).unsqueeze(0)
plt.figure()
#im_opt = np.reshape(x_opt, (size,size))
plt.imshow(im_opt)
plt.show()
