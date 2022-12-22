import sys
if __name__ == "__main__":
    sys.path.append("../pyscatwave")
from os.path import join
from math import sqrt
import numpy as np
import scipy as sp
import scipy.optimize as opt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, grad
from termcolor import colored
import complex_utils as cplx
from metric import PhaseHarmonicCov, PhaseHarmonicCovNonLinCoeff, PhaseHarmonicPruned
import make_figs as signal
from time import time
from utils import cuda_available, make_dir_if_not_there
from itertools import product
from global_const import RESPATH, DATAPATH
from tqdm import tqdm
from loss import PSNR
import librosa.core


class SolverHack(nn.Module):
    def __init__(self, embedding, x0, loss_fn, M=None, cuda=False):
        super(SolverHack, self).__init__()

        self.embedding = embedding
        self.x0 = x0
        self.loss = loss_fn
        self.M = M
        self.is_cuda = False

        self.last_x = None
        self.last_x_torch = None
        self.last_emb = None
        self.last_err = None

        if cuda:
            self.cuda()
        x0_torch = self.format(self.x0, requires_grad=False)
        # x0_torch = torch.stack((x0_torch, torch.zeros_like(x0_torch)), dim=-1)
        self.emb_fst0 = self.embedding.compute_first_order(x0_torch)

        if isinstance(self.embedding, PhaseHarmonicCovNonLinCoeff):
            self.embedding.non_linear_coefficient_selection(x0_torch, self.emb_fst0)

        self.emb0 = self.embedding(x0_torch, self.emb_fst0)

        self.set_coeff_to_zero()
        self.err0 = self.loss(None, self.emb0)

    def cuda(self):
        if not self.is_cuda:
            self.is_cuda = True
            self.embedding = self.embedding.cuda()
            if 'emb0' in self.__dict__:
                self.emb0 = self.emb0.cuda()
                self.emb_fst0 = self.emb_fst0.cuda()
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
            if 'emb0' in self.__dict__:
                self.emb0 = self.emb0.cpu()
                self.emb_fst0 = self.emb_fst0.cpu()
            if self.last_emb is not None:
                self.last_emb = self.last_emb.cpu()
            if self.last_x_torch is not None:
                self.last_x_torch = self.last_x_torch.cpu()
            if self.last_err is not None:
                self.last_err = self.last_err.cpu()
        return self

    def set_coeff_to_zero(self):
        if len(self.emb0) == 2:
            emb0_fst, emb0_scd = self.emb0
            num_fst = emb0_fst.size(1)
        else:
            emb0_scd, = self.emb0
            num_fst = 0

        energy = (cplx.modulus(emb0_scd) ** 2).data.cpu().numpy()[0]
        if self.M is None:
            self.energy_ratio = 1.
            self.num_coeff = energy.size + num_fst
        else:
            num_idx_keep = int(self.M * energy.size)
            min_idx = np.argsort(energy)[:-num_idx_keep]

            drop_energy = energy[min_idx]
            tqdm.write("M={}: dropped {:.2E}% of embedding energy".format(
                self.M, round(100 * np.sum(drop_energy) / np.sum(energy), 2)
                ))
            self.energy_ratio = 1 - np.sum(drop_energy) / np.sum(energy)
            self.num_coeff = num_idx_keep + num_fst
            emb0_scd[:, min_idx] = 0


        if len(self.emb0) == 2:
            self.emb0 = emb0_fst, emb0_scd
        else:
            self.emb0 = emb0_scd,

    def format(self, x, requires_grad=True):
        """Transforms x into a compatible format for the embedding."""

        x = cplx.from_numpy(x[None, None], tensor=torch.DoubleTensor)
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
            self.last_emb = self.embedding(self.last_x_torch, self.emb_fst0)

            self.last_err = self.loss(self.last_emb, self.emb0) / self.err0

        return float(self.last_err.data.cpu().numpy())


    def grad_function(self, x):
        """returns gradient of 'function'."""

        err = self.function(x)
        grad_err, = grad([self.last_err], [self.last_x_torch], retain_graph=True)

        # only get the real part
        grad_err = grad_err.data.cpu().numpy()[0, 0, :, 0]

        return grad_err

    def joint(self, x):
        f = self.function(x)
        gf = np.array(self.grad_function(x).ravel(), dtype=float)
        return f, gf

    def grad_err(self, x):
        g = self.grad_function(x)
        return np.linalg.norm(g, ord=float('inf'))

def mse_loss(emb, emb0):
    if len(emb0) == 2:
        emb0_fst, emb0_scd = emb0
        if emb is None:
            emb_fst = torch.zeros_like(emb0_fst)
            emb_scd = torch.zeros_like(emb0_scd)
        else:
            emb_fst, emb_scd = emb

        err_fst = F.mse_loss(emb_fst, emb0_fst)
        err_scd = F.mse_loss(emb_scd, emb0_scd)

        return err_fst ** 2 + err_scd

    else:
        emb0_scd, = emb0
        if emb is None:
            emb_scd = torch.zeros_like(emb0_scd)
        else:
            emb_scd, = emb
        err_scd = F.mse_loss(emb_scd, emb0_scd)
        return err_scd


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
        f_gap0 = sel(f_gap, 1, self.phi.xi_idx[:,  0])
        f_gap1 = sel(f_gap, 1, self.phi.xi_idx[:,  1])
        t_f0 = sel(t_f, 1, self.phi.xi_idx[:, 0])
        t_f1 = sel(t_f, 1, self.phi.xi_idx[:, 1])
        err_fst0 = cplx.mul(f_gap0, cplx.conjugate(t_f1))
        err_fst1 = cplx.mul(t_f0, cplx.conjugate(f_gap1))

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


def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60.
    return "{}:{:>02}:{:>05.2f}".format(h, m, s)


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
            tqdm.write(colored("{:6}it in {} ( {:.2f}it/s )  ........  {:.3E} -- {:.3E}".format(
                self.counter, hms_string(delta_t), self.counter / delta_t,
                err, gerr
                ), 'blue'))
            self.next_milestone /= 10


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    n_experiment = 1
    seeds = [np.random.randint(2 ** 20) for _ in range(n_experiment)]
    maxiter = 5000
    max_chunk = 2000
    # data = 'lena'
    data = 'lena'
    # data = 'local_smooth'
    # data = 'single_freq_modulated_bis'

    do_cuda = False
    print()
    cuda = cuda_available()
    if do_cuda or not cuda:
        print("CUDA available: {}\n".format(cuda))
    else:
        print("CUDA denied\n".format(cuda))
        cuda = False

    line_idx = 224
    set_to_zero = True

    nscales_l = [4]
    Qs = [1]
    # Qs = [8]
    nocts = [2]
    # nocts = [4]
    K_mults = [1.0]
    # K_mults = [1.]
    # k_types = ['linear', 'log2']
    k_types = ['log2']
    # wavelet_types = ["bump_steerable", "battle_lemarie"]
    # wavelet_types = ["battle_lemarie"]
    wavelet_types = ["morlet"]
    fst_orders = [True]
    tol = 1.2
    high_freq = 0.5
    multi_same = False
    Ms = [None]  # number or fraction of second order coefficients kept, keep all if None
    # Ms = [1.4 ** (-i) for i in range(20)]  # number or fraction of second order coefficients kept

    T = 1024
    if data == 'lena':
        x0_raw, line_idx = signal.lena_line(line_idx=line_idx)
        extra_info = {'line_idx': line_idx}
        save_dir = data + '_ktype_{}'.format(line_idx)
    elif data == 'data':
        # filename = 'applause_2.0s_8192.wav'
        filename = 'flute_2.0s_8192.wav'
        load_path = join(DATAPATH, "gen_phaseexp_inv", filename)
        x0_raw, _ = librosa.core.load(load_path)
        rate = int(filename.split('.')[-2].split('_')[-1])
        extra_info = {'sr': rate}
        print(colored("\nSignal info: size {} at rate {}".format(x0_raw.shape, rate), 'red'))
        x0_raw = torch.Tensor(x0_raw[None, None, :])
        save_dir = '.'.join(filename.split('.')[:-1])
    elif data == 'local_smooth':
        n_set = 6
        x0_raw = signal.locally_smooth(T, n_set)
        extra_info = {'n_set': n_set}
        save_dir = 'locallysmooth{}'.format(n_set)
    elif data == 'single_freq_modulated':
        per0, per1 = 11., 127.
        x0_raw = signal.single_freq_modulated(T, per0=per0, per1=per1)
        extra_info = {'per0': per0, 'per1': per1}
        save_dir = 'single_freq_modulated'
    elif data == 'single_freq_modulated_bis':
        per0, per1 = 5., 127.
        x0_raw = signal.single_freq_modulated_bis(T, per0=per0, per1=per1)
        extra_info = {'per0': per0, 'per1': per1}
        save_dir = 'single_freq_modulated_bis'
    else:
        raise ValueError("Unknown data: '{}'".format(data))

    # plt.figure()
    # plt.plot(x0_raw.data.cpu().numpy()[0, 0])
    # plt.show()

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
                    x0 = x0_raw.clone()

                    signal_info = data
                    x0 = x0.cpu().numpy()[0, 0]
                    x = np.random.randn(*x0.shape) / np.sqrt(x0.size)
                    T = x0.shape[-1]

                    ndiag = noct * Q + 1
                    if set_to_zero:  # set coefficients to 0
                        phi = PhaseHarmonicCov(
                            nscales, Q, K, T, k_type=k_type, wav_type=wavelet_type,
                            ndiag=ndiag, fst_order=fst_order, tolerance=tol,
                            multi_same=multi_same, high_freq=high_freq,
                            check_for_nan=False, max_chunk=max_chunk)
                    else:  # ignore coefficients
                        phi = NonLinCoeff(
                            nscales, Q, K, T, M=M, k_type=k_type, wav_type=wavelet_type,
                            ndiag=ndiag, fst_order=fst_order, tolerance=tol,
                            multi_same=multi_same, high_freq=high_freq,
                            check_for_nan=False, max_chunk=max_chunk)
                    # loss_fn = mse_loss
                    loss_fn = MSELoss(phi)

                    if set_to_zero:
                        function_obj = SolverHack(phi, x0, loss_fn, M=M, cuda=cuda)
                        num_coeff = function_obj.num_coeff
                    else:
                        function_obj = SolverHack(phi, x0, loss_fn, M=None, cuda=cuda)
                        num_coeff = phi.shape_effect()

                    check_conv_criterion = CheckConvCriterion(function_obj, 1e-12)

                    tqdm.write(colored(emb_descr, 'red'))
                    tqdm.write(colored("Using embedding " + function_obj.embedding.__class__.__name__, 'red'))
                    tqdm.write(colored("Embedding using {} coefficients.".format(num_coeff), 'red'))
                    if isinstance(phi, PhaseHarmonicCov):
                        energy_ratio = function_obj.energy_ratio
                    else:
                        energy_ratio = phi.energy_ratio

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
                        'num_coeff': num_coeff, 'M': M, 'energy_ratio': energy_ratio,
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
