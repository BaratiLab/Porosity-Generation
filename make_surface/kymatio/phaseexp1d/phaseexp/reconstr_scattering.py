import sys
if __name__ == "__main__":
    sys.path.append("../pyscatwave")
from os.path import join
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.optimize as opt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, grad
from termcolor import colored
import complex_utils as cplx
from metric import PhaseHarmonicCov, PhaseHarmonicCovNonLinCoeff
import make_figs as signal
from time import time
from utils import cuda_available, make_dir_if_not_there
from itertools import product
from global_const import RESPATH, DATAPATH
from tqdm import tqdm
from loss import PSNR
import librosa.core
from scatwave.scattering1d.scattering1d import Scattering1D


class SolverHack(nn.Module):
    def __init__(self, embedding, x0, loss_fn, cuda=False):
        super(SolverHack, self).__init__()

        self.embedding = embedding
        self.x0 = x0
        self.loss = loss_fn
        self.is_cuda = False

        self.last_x = None
        self.last_x_torch = None
        self.last_emb = None
        self.last_err = None


        if cuda:
            self.cuda()
        x0_torch = self.format(self.x0, requires_grad=False)
        # x0_torch = torch.stack((x0_torch, torch.zeros_like(x0_torch)), dim=-1)
        self.emb0 = self.embedding(x0_torch)
        self.err0 = self.loss(None, self.emb0)
        self.num_coeff = self.emb0.size(1)

    def cuda(self):
        if not self.is_cuda:
            self.is_cuda = True
            self.embedding = self.embedding.cuda()
            if self.last_emb is not None:
                self.last_emb = self.last_emb.cuda()
            if self.last_x_torch is not None:
                self.last_x_torch = self.last_x_torch.cuda()
            if self.last_err is not None:
                self.last_err = self.last_err.cuda()
        return self

    def cpu(self):
        if self.is_cuda:
            self.is_cuda = False
            self.embedding = self.embedding.cpu()
            if self.last_emb is not None:
                self.last_emb = self.last_emb.cpu()
            if self.last_x_torch is not None:
                self.last_x_torch = self.last_x_torch.cpu()
            if self.last_err is not None:
                self.last_err = self.last_err.cpu()
        return self

    def format(self, x, requires_grad=True):
        """Transforms x into a compatible format for the embedding."""

        x = torch.Tensor(x[None, None])
        x = Variable(x, requires_grad=requires_grad)
        if self.is_cuda:
            x = x.cuda()
        return x

    def function(self, x):
        """returns L(phi(x), phi(x0)) / L(0, phi(x0))."""

        if self.last_x is None or not np.all(np.equal(self.last_x, x)):
            self.last_x = x.copy()
            x_torch = self.format(self.last_x)
            self.last_x_torch = x_torch
            # x_torch = torch.stack((x_torch, torch.zeros_like(x_torch)), dim=-1)

            if self.last_x_torch.grad is not None:
                self.last_x_torch.grad.data.zero_()
            self.last_emb = self.embedding(self.last_x_torch)

            self.last_err = self.loss(self.last_emb, self.emb0) / self.err0

        return float(self.last_err.data.cpu().numpy())


    def grad_function(self, x):
        """returns gradient of 'function'."""

        err = self.function(x)
        grad_err, = grad([self.last_err], [self.last_x_torch], retain_graph=True)

        # only get the real part
        grad_err = grad_err.data.cpu().numpy()[0, 0, :]

        return grad_err

    def joint(self, x):
        f = self.function(x)
        gf = np.array(self.grad_function(x).ravel(), dtype=float)
        return f, gf

    def grad_err(self, x):
        g = self.grad_function(x)
        return np.linalg.norm(g, ord=float('inf'))

def mse_loss(emb, emb0):
    if emb is None:
        emb = torch.zeros_like(emb0)
    err = F.mse_loss(emb, emb0)
    return err


class MSELoss(nn.Module):
    def __init__(self, phi):
        super(MSELoss, self).__init__()
        self.phi = phi

    def scd_order_mse(self, input, target):
        target_scd, = target
        if input is None:
            input_scd = torch.zeros_like(target_scd)
        else:
            input_scd, = input
        return self.mse_norm(input_scd - target_scd)

    def fst_order_mse(self, input, target):
        t_f, t_s = target
        if input is None:
            i_f = torch.zeros_like(t_f)
            i_s = torch.zeros_like(t_s)
        else:
            i_f, i_s = input

        s_gap = i_s - t_s
        f_gap = i_f - t_f

        sel = torch.index_select
        err_fst0 = sel(f_gap, 1, self.phi.xi_idx[:,  0]) * sel(t_f, 1, self.phi.xi_idx[:, 1])
        err_fst1 = sel(f_gap, 1, self.phi.xi_idx[:,  1]) * sel(t_f, 1, self.phi.xi_idx[:, 0])

        gap = s_gap - err_fst0 - err_fst1
        return self.mse_norm(gap)

    def forward(self, input, target):
        if self.phi.fst_order:
            return self.fst_order_mse(input, target)
        else:
            return self.scd_order_mse(input, target)

    @staticmethod
    def mse_norm(x):
        sq_err = cplx.modulus(x) ** 2
        return torch.mean(sq_err.view(-1))


class SmallEnoughException(Exception):
    pass


class CheckConvCriterion:
    def __init__(self, phi, tol):
        super(CheckConvCriterion, self).__init__()
        self.phi = phi
        self.tol = tol
        self.result = None
        self.next_milestone = None
        self.counter = 0
        self.err = None
        self.gerr = None
        self.tic = time()

    def __call__(self, xk):
        err = self.phi.function(xk)
        gerr = np.linalg.norm(self.phi.grad_function(xk), ord=float('inf'))
        self.counter += 1
        self.err = err
        self.gerr = gerr

        if self.next_milestone is None:
            self.next_milestone = 10 ** (np.floor(np.log10(gerr)))

        if err <= self.tol:
            self.result = xk
            raise SmallEnoughException()
        elif gerr <= self.next_milestone:
            delta_t = time() - self.tic
            tqdm.write(colored("{:6}it in {:6}s ( {:.2f}it/s )  ........  {:.3E} -- {:.3E}".format(
                self.counter, round(delta_t, 2), self.counter / delta_t,
                err, gerr
                ), 'blue'))
            self.next_milestone /= 10


class Scattering(Scattering1D):
    def __init__(self, T, J, Q, **kwargs):
        super(Scattering, self).__init__(T, J, Q, **kwargs)

    def forward(self, x, **kwargs):
        phi_x = super(Scattering, self).forward(x, **kwargs)
        phi_x = phi_x.mean(dim=2)
        return phi_x


if __name__ == "__main__":

    n_experiment = 1
    seeds = [np.random.randint(2 ** 20) for _ in range(n_experiment)]
    maxiter = 5000
    max_chunk = 2000
    # data = 'lena'
    data = 'data'

    do_cuda = True
    print()
    cuda = cuda_available()
    if do_cuda or not cuda:
        print("CUDA available: {}\n".format(cuda))
    else:
        print("CUDA denied\n".format(cuda))
        cuda = False

    line_idx = 224
    set_to_zero = True

    scattering_kwargs = {
        'normalize': 'l1',

    }

    nscales_l = [8]
    Qs = [24]
    # Qs = [8]
    nocts = [2]
    # nocts = [4]
    K_mults = [1.0]
    # K_mults = [1.]
    # k_types = ['linear', 'log2']
    k_types = ['log2']
    # wavelet_types = ["bump_steerable", "battle_lemarie"]
    wavelet_types = ["battle_lemarie"]
    fst_orders = [True]
    tol = 1.2
    high_freq = 0.5
    multi_same = False
    Ms = [None]  # number or fraction of second order coefficients kept, keep all if None
    # Ms = [1.4 ** (-i) for i in range(20)]  # number or fraction of second order coefficients kept

    params_order = ('J', 'Q', 'K', 'M', 'Jmax', 'fst_order', 'k_type', 'wav_type')
    for num_exp, seed in enumerate(seeds):
        set_done = set()
        args = list(product(nscales_l, Qs, K_mults, Ms, nocts, fst_orders, k_types, wavelet_types))
        exp_desc = 'Experiment {}/{}'.format(num_exp + 1, len(seeds))
        with tqdm(args, desc=colored(exp_desc, 'yellow')) as t:
            for nscales, Q, K_mult, M, noct, fst_order, k_type, wavelet_type in t:

                if k_type == 'linear':
                    K_basis = 2 ** noct + 1
                elif k_type == 'log2':
                    K_basis = max(3, noct + 1)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
                K = int(K_basis * K_mult)
                params = (nscales, Q, K, M, noct, fst_order, k_type, wavelet_type)
                emb_descr = "Embedding: " + ' '.join(
                    ['{}={}'.format(name, val) for name, val in zip(params_order, params)])

                if K < 3 or params in set_done:
                    print(colored(emb_descr + " : PASS", 'red'))
                else:
                    set_done.add(params)

                    # set random seed
                    tqdm.write('Random seed used : {}'.format(seed))
                    np.random.seed(seed)
                    torch.manual_seed(seed + 1)
                    torch.cuda.manual_seed(seed + 2)

                    # generate data
                    if data == 'lena':
                        x0, line_idx = signal.lena_line(line_idx=line_idx)
                        extra_info = {'line_idx': line_idx}
                        save_dir = data + '_ktype_{}'.format(line_idx)
                    elif data == 'data':
                        filename = 'flute_0.8s_8192.wav'
                        load_path = join(DATAPATH, "gen_phaseexp_inv", filename)
                        x0, _ = librosa.core.load(load_path)
                        rate = int(filename.split('.')[-2].split('_')[-1])
                        extra_info = {'sr': rate}
                        print(colored("\nSignal info: size {} at rate {}".format(x0.shape, rate), 'red'))
                        x0 = torch.Tensor(x0[None, None, :])
                        save_dir = '.'.join(filename.split('.')[:-1])
                    else:
                        raise ValueError("Unknown data: '{}'".format(data))
                    save_dir = save_dir + "_scattering"
                    signal_info = data
                    x0 = x0.cpu().numpy()[0, 0]
                    x = np.random.randn(*x0.shape) / np.sqrt(x0.size)
                    T = x0.shape[-1]

                    ndiag = noct * Q + 1
                    phi = Scattering(T, nscales, Q)
                    # loss_fn = mse_loss
                    loss_fn = mse_loss

                    if set_to_zero:
                        function_obj = SolverHack(phi, x0, loss_fn, cuda=cuda)
                        num_coeff = function_obj.num_coeff
                    else:
                        function_obj = SolverHack(phi, x0, loss_fn, cuda=cuda)
                        num_coeff = phi.shape_effect()

                    check_conv_criterion = CheckConvCriterion(function_obj, 1e-12)

                    tqdm.write(colored(emb_descr, 'red'))
                    tqdm.write(colored("Using embedding " + function_obj.embedding.__class__.__name__, 'red'))
                    tqdm.write(colored("Embedding using {} coefficients.".format(num_coeff), 'red'))

                    wavelet_info = wavelet_type.replace("_", "-") + '{:.3f}'.format(high_freq)
                    save_name = signal_info + "_{}o2_{}-{}coeff_{}_{}_N{}_Q{}_K{}_M{}_seed{}".format(
                        "o1" if fst_order else "", 'tol{:.2f}'.format(tol), num_coeff,
                        wavelet_info, 'MSE', nscales, Q, K, M, seed)
                    tqdm.write(save_name)


                    tic = time()

                    method = 'L-BFGS-B'
                    # method = 'CG'
                    try:
                        jac = True
                        func = function_obj.joint if jac else function_obj.function
                        res = opt.minimize(
                            func, x, method=method, jac=jac, tol=1e-12,
                            callback=check_conv_criterion, options={'maxiter': maxiter}
                        )
                        final_loss, x_opt, niter, msg = res['fun'], res['x'], res['nit'], res['message']
                    except SmallEnoughException:
                        x_opt = check_conv_criterion.result
                        niter = check_conv_criterion.counter
                        msg = "SmallEnoughException"

                    toc = time()

                    tqdm.write(colored("    ----    ", 'blue'))

                    if not isinstance(msg, str):
                        msg = msg.decode("ASCII")
                    final_loss = function_obj.function(x_opt)
                    final_gloss = function_obj.grad_err(x_opt)
                    tqdm.write(colored('Optimization Exit Message : ' + msg, 'blue'))
                    tqdm.write(colored("found parameters in {}s, {} iterations -- {}it/s".format(
                        round(toc - tic, 4), niter, round(niter / (toc - tic), 2)), 'blue'))
                    tqdm.write(colored("    relative error {:.3E}".format(final_loss), 'blue'))
                    tqdm.write(colored("    relative gradient error {:.3E}".format(final_gloss), 'blue'))
                    tqdm.write(colored("    x0 norm {:.3E}".format(float(function_obj.err0.data.cpu().numpy())), 'blue'))

                    offset = signal.offset_greed_search(x_opt, x0)
                    x_opt = np.roll(x_opt, offset)

                    # err = np.sqrt(np.mean((x_opt - x0) ** 2) / np.mean(x0 ** 2))
                    # tqdm.write("L2 error : {}".format(err))
                    err = PSNR(x_opt, x0, 2.)  # signal values are always in [-1, 1]
                    tqdm.write(colored("PSNR : {}".format(err), 'green'))


                    save_path = join(RESPATH, "gen_phaseexp_inv/", save_dir)
                    make_dir_if_not_there(save_path)
                    save_var = {
                        'x0': x0, 'x': x_opt, 'err': err, 'order': 2, 'seed': seed,
                        'N': nscales, 'Q': Q, 'K': K, 'M': M, 'noct': noct, 'T': T, 'tol': tol,
                        'fst_order': fst_order, 'wav_type': wavelet_type, 'k_type': k_type,
                        'multi_same': multi_same, 'final_loss': final_loss,            
                        'num_coeff': num_coeff,
                        **extra_info
                    }
                    npz_path = join(save_path, save_name + ".npz")
                    np.savez(npz_path, **save_var)
                    tqdm.write("save as '{}'\n\n".format(npz_path))

                    # plt.figure(figsize=(12, 12))
                    # ax = plt.subplot2grid((4, 1), (0, 0), rowspan=3)
                    # ax.plot(x0, 'r')
                    # ax.plot(x_opt, 'b')
                    # ax.set_xticklabels([])
                    # ax = plt.subplot2grid((4, 1), (3, 0), rowspan=1)
                    # ax.plot(x_opt - x0, 'r')
                    # # plt.show()
                    # plt.savefig(join(save_path, save_name + '.pdf'))

                    tqdm.write("\n    ----------------------------------------\n")
