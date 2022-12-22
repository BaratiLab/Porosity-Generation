import sys
if __name__ == "__main__":
    sys.path.append ("../pyscatwave_debug/pyscatwave/")
    sys.path.append ("../phaseexp")
import os.path
import numpy as np
import scipy as sp
import scipy.io
import scipy.optimize
import optim
import torch
import torch.autograd
import complex_utils as cplx
from metric import PhaseHarmonicPruned
from global_consts import DATA_PATH, SAVE_PATH, Tensor
from solver_hack_phase import SolverHack, MSELoss, CheckConvCriterion, SmallEnoughException
from utils import make_dir_if_not_there, cuda_available
import make_figs as signal
from loss import PSNR
#import matplotlib.pyplot as plt
from time import time
from tqdm import tqdm, trange
from itertools import product
from termcolor import colored

torch.manual_seed(0)

cuda_flag = cuda_available()

signal_choice = 'staircase'

# Optimization
maxiter = 5000
tol_optim = 1e-10

# Embedding parameters
T = 2**10
nscales = 9
Q = 1
deltaj = 1
wavelet_type = 'battle_lemarie'

if signal_choice == 'staircase':
    x0 = signal.staircase(T, 10).squeeze(1).squeeze(0)

#x0_torch = cplx.from_numpy(x0[None,None], tensor=torch.DoubleTensor)
#x0_torch = torch.autograd.Variable(x0_torch, requires_grad=False)
x0_torch = x0[None, None]
x0_torch.requires_grad = False

phi = PhaseHarmonicPruned(
    nscales, Q, T, wav_type=wavelet_type,
    delta_j = deltaj)

loss_fn = MSELoss(phi)
#optimizer = optim.LBFGSDescent(loss_type=loss_fn)
optimizer = optim.SGDDescent(loss_type=loss_fn)

if cuda_flag:
    phi=phi.cuda()
    x0_torch = x0_torch.cuda()
    loss_fn = loss_fn.cuda()
    optimizer = optimizer.cuda()

print("Step1: optimization with Torch's solver")
print('------')
tic = time()
res = optimizer(x0_torch, phi, maxiter=maxiter, tol=tol_optim)
timet1 = time() - tic
niter1, loss1, msg1 = res['niter'], res['loss'], res['msg']
# x1 = res['x'].cpu().detach().numpy().squeeze(1).squeeze(0)

# ## Repeat with scipy's  solver ##
# print("Step2: optimization with Torch's solver")
# print('------')

# print(x0.shape)

# # Recreate all objects to avoid memory effects
# phi = PhaseHarmonicPruned(
#     nscales, Q, T, wav_type=wavelet_type,
#     delta_j = deltaj)
# loss_fn = MSELoss(phi)
# solver_fn = SolverHack(phi, x0, loss_fn, cuda=cuda_flag)
# if cuda_flag:
#     phi=phi.cuda()
#     loss_fn = loss_fn.cuda()
#     solver_fn = solver_fn.cuda()

# xini = np.random.randn(*x0.shape)
# check_conv_criterion = CheckConvCriterion(solver_fn, 1e-12)
# jac = True
# func = solver_fn.joint if jac else solver_fn.function
# tic = time()
# try:
#     res = scipy.optimize.minimize(
#         solver_fn.joint, xini, method='L-BFGS-B', jac=jac, tol=tol_optim,
#         callback=check_conv_criterion,
#         options={'maxiter': maxiter})
#     x2, niter2, loss2, msg2 = res['x'], res['nit'], res['fun'], res['message']
# except SmallEnoughException:
#     print('Finished through SmallEnoughException')
# timet2 = time() - tic

# print('')
# print('torch.optim:')
# print('Time: {:6} s, Loss: {:.6E}, niter: {:5}'.format(timet1, loss1, niter1))
# print('scipy.optim:')
# print('Time: {:6} s, Loss: {:.6E}, niter: {:5}'.format(timet2, loss2, niter2))

# ## Comment out this region if running remotely on iphyton without X forwarding.

# import matplotlib.pyplot as plt

# plt.figure()
# plt.subplot(311)
# plt.plot(x0.cpu().numpy())
# plt.subplot(312)
# plt.plot(x1)
# plt.subplot(313)
# plt.plot(x2)
# plt.show()
