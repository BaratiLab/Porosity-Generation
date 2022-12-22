import sys
sys.path.append("../pyscatwave/")
import numpy as np
import scipy as sp
import scipy.io
import scipy.optimize
import torch
from torch.autograd import Variable

from full_embedding import FullEmbedding
from solver_hack_full import SolverHack, MSELoss, offset_greed_search_psnr
from check_conv_criterion import CheckConvCriterion, SmallEnoughException

from utils import cuda_available
import make_figs as signal
from loss import PSNR
import matplotlib.pyplot as plt
from time import time
from tqdm import tqdm, trange
from itertools import product
from termcolor import colored

do_cuda = True  # if True, will check if CUDA is available et use the GPU accordingly

print()
cuda = cuda_available()
if do_cuda or not cuda:
    print("CUDA available: {}\n".format(cuda))
else:
    print("CUDA denied\n".format(cuda))
    cuda = False


#----- Analysis parameters -----
T = 2**14
J = (9)
Q = 10

# Select which scattering orders and phase harmonic coefficients are used
scatt_orders = (1, 2)
phe_coeffs = ('harmonic', 'mixed')

# Parameters for phase harmonic coefficients
deltaj = 3
deltak = (0,)
num_k_modulus = 0
delta_cooc = 2

wavelet_type = 'morlet'
high_freq = 0.425   # Center frequency of mother wavelet

max_chunk = 2 # None #  50 # None  # Process coefficients in chunks of size max_chunk if too much memory is used
alpha = 0.5  # weight of scattering in loss, phase harmonic has weight 1 - alpha

maxiter = 1000
tol = 1e-12

#----- Create data -----

n_dirac = 10  # number of discontinuities
x0 = signal.staircase(T, n_dirac)  # Created as torch tensor


# Convert to numpy and create random intialization
x0 = x0.cpu().numpy()[0, 0]

#----- Setup Analysis -----

# Pack parameters for phase harmonics and scattering
phe_params = {
    'delta_j': deltaj, 'delta_k': deltak,
    'wav_type':wavelet_type, 'high_freq':high_freq,
    'delta_cooc': delta_cooc, 'max_chunk': max_chunk
    }
scatt_params = dict()


# Create full embedding that combines scattering and phase harmonics
phi = FullEmbedding(
    T, J, Q, phe_params=phe_params, scatt_params=scatt_params,
    scatt_orders=scatt_orders, phe_coeffs=phe_coeffs)
num_coeff, nscat = phi.shape(), phi.count_scatt_coeffs()
nharm = num_coeff - nscat

# Create loss object
loss_fn = MSELoss(phi, alpha=alpha, use_cuda=cuda)

#  Create solver object to interface embeddings with scipy's optimize
solver_fn = SolverHack(phi, x0, loss_fn, cuda=cuda)


if cuda:
     phi=phi.cuda()
     loss_fn = loss_fn.cuda()
     solver_fn = solver_fn.cuda()

# Create object that checks and prints convergence information
check_conv_criterion = CheckConvCriterion(solver_fn, 1e-24)

# Decide how to compute the gradient
jac = True   # True: provided by solver_fn, False: computed numerically by minimize
func = solver_fn.joint if jac else solver_fn.function


#----- Optimization -----

# Initial point
xini = np.random.randn(*x0[None, None].shape)

tic = time()
try:
    options = {'maxiter': maxiter, 'maxfun': maxiter}
    res = sp.optimize.minimize(
        solver_fn.joint, xini, method='L-BFGS-B', jac=jac,
        callback=check_conv_criterion, tol=tol,
        options=options)
    x, niter, loss, msg = res['x'], res['nit'], res['fun'], res['message']
except SmallEnoughException:
    print('Finished through SmallEnoughException')
toc = time()

# Recover final loss
final_loss, final_grad = solver_fn.joint(x)
final_gloss = np.linalg.norm(final_grad, ord=float('inf'))

if not isinstance(msg, str):
    msg = msg.decode("ASCII")

print(colored('Optimization Exit Message : ' + msg, 'blue'))
print(colored("found parameters in {}s, {} iterations -- {}it/s".format(
    round(toc - tic, 4), niter, round(niter / (toc - tic), 2)), 'blue'))
print(colored("    relative error {:.3E}".format(final_loss), 'blue'))
print(colored("    relative gradient error {:.3E}".format(final_gloss), 'blue'))
x0_norm_msg = "    x0 norm  S{:.2E}  H{:.2E}".format(
    float(solver_fn.loss_scat0.data.cpu().numpy()),
    float(solver_fn.loss_scat0.data.cpu().numpy())
    )
print(colored(x0_norm_msg, 'blue'))

# Recover log of loss throughout optimization
logs_loss = check_conv_criterion.logs_loss
logs_grad = check_conv_criterion.logs_grad
logs_scat = check_conv_criterion.logs_scat
logs_harm = check_conv_criterion.logs_harm

# Recenter data and compute PSNR:
offset = offset_greed_search_psnr(x, x0)
x = np.roll(x, offset)


#----- Plot results -----

plt.figure()
plt.subplot(211)
plt.plot(x0)
plt.subplot(212)
plt.plot(x)
plt.show()
