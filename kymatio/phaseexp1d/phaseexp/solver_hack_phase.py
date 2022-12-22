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
    def __init__(self, embedding, x0, loss_fn, cuda=False):
        super(SolverHack, self).__init__()

        self.embedding = embedding
        self.x0 = x0
        self.loss = loss_fn
        self.is_cuda = False
        self.res = None, None

        if cuda:
            self.cuda()

        # compute embedding and loss at initial guess
        x0_torch = self.format(self.x0, requires_grad=False)
        self.emb0 = self.embedding(x0_torch)
        # print("emb0 shape:", self.emb0[0].shape, self.emb0[1].shape)
        self.err0 = self.loss(self.emb0, None)

    def cuda(self):
        if not self.is_cuda:
            self.is_cuda = True
            self.embedding = self.embedding.cuda()
            if 'emb0' in self.__dict__:
                self.emb0 = self.emb0.cuda()
        return self

    def cpu(self):
        if self.is_cuda:
            self.is_cuda = False
            self.embedding = self.embedding.cpu()
            if 'emb0' in self.__dict__:
                self.emb0 = self.emb0.cpu()
        return self

    def format(self, x, requires_grad=True):
        """Transforms x into a compatible format for the embedding."""

        x = cplx.from_numpy(x[None, None], tensor=torch.DoubleTensor)
        if self.is_cuda:
            x = x.cuda()
        x = Variable(x, requires_grad=requires_grad)
        return x

    def joint(self, x):
        # format x and set gradient to 0
        x_torch = self.format(x)
        if x_torch.grad is not None:
            x_torch.grad.data.zero_()

        # compute embedding
        emb = self.embedding(x_torch)

        # compute loss function
        loss = self.loss(emb, self.emb0) / self.err0

        # compute gradient
        grad_x, = grad([loss], [x_torch], retain_graph=True)

        # only get the real part
        grad_x = grad_x[0, 0, ..., 0]

        # move to numpy
        grad_x = grad_x.contiguous().detach().data.cpu().numpy()
        loss = loss.detach().data.cpu().numpy()

        self.res = loss, grad_x

        return loss, grad_x


class MSELoss(nn.Module):
    def __init__(self, phi):
        super(MSELoss, self).__init__()
        self.phi = phi

    def forward(self, input, target):
        i_f, i_s = input
        if target is None:
            t_f = torch.zeros_like(i_f)
            t_s = torch.zeros_like(i_s)
        else:
            t_f, t_s = target

        s_gap = i_s - t_s
        f_gap = i_f - t_f

        sel = torch.index_select

        gap = []
        start = 0
        for xi_idx, ks in zip(self.phi.xi_idx, self.phi.ks):
            f_gap0 = sel(f_gap, 2, xi_idx[:,  0])
            f_gap1 = sel(f_gap, 2, xi_idx[:,  1])
            t_f0 = sel(t_f, 2, xi_idx[:, 0])
            t_f1 = sel(t_f, 2, xi_idx[:, 1])

            # set to zero first order coefficients with harmonic k > 0
            idx_null0 = (ks[..., 0] > 0).unsqueeze(0).unsqueeze(0).unsqueeze(-1)
            idx_null1 = (ks[..., 1] > 0).unsqueeze(0).unsqueeze(0).unsqueeze(-1)
            f_gap0 = f_gap0.masked_fill(idx_null0, 0)
            f_gap1 = f_gap1.masked_fill(idx_null1, 0)
            t_f0 = t_f0.masked_fill(idx_null0, 0)
            t_f1 = t_f1.masked_fill(idx_null1, 0)

            err_fst0 = cplx.mul(f_gap0, cplx.conjugate(t_f1))
            err_fst1 = cplx.mul(t_f0, cplx.conjugate(f_gap1))

            l = xi_idx.size(0)
            s_gap_l = s_gap[:, :, start:start + l]
            start += l

            g = s_gap_l - err_fst0 - err_fst1
            gap.append(g)
        gap = torch.cat(gap, dim=2)
        return self.mse_norm(gap)

    @staticmethod
    def mse_norm(x):
        sq_err = (x ** 2).sum(-1)
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
        self.logs = []
        self.glogs = []

    def __call__(self, xk):
        # err, grad_xk = self.phi.joint(xk)
        err, grad_xk = self.phi.res
        gerr = np.linalg.norm(grad_xk, ord=float('inf'))
        self.logs.append(float(err))
        self.glogs.append(float(gerr))
        self.err = err
        self.gerr = gerr
        self.counter += 1

        if self.next_milestone is None:
            # self.next_milestone = 10 ** (np.floor(np.log10(gerr)))
            self.next_milestone = gerr

        if err <= self.tol:
            self.result = xk
            raise SmallEnoughException()
        elif gerr <= self.next_milestone:
            delta_t = time() - self.tic
            tqdm.write(colored("{:6}it in {} ( {:.2f}it/s )  ........  {:.3E} -- {:.3E}".format(
                self.counter, hms_string(delta_t), self.counter / delta_t,
                err, gerr
                ), 'blue'))
            self.next_milestone /= 2.
            # self.next_milestone /= 10


def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60.
    return "{}:{:>02}:{:>05.2f}".format(h, m, s)


def offset_greed_search_psnr(x, x0):
    T = np.size(x)
    max_psnr, best_offset = float('-inf'), None
    for offset in range(T):
        x1 = np.roll(x, offset)
        psnr = PSNR(x1, x0, 2.)
        if psnr > max_psnr:
            max_psnr, best_offset = psnr, offset

    return best_offset


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    n_exp = 1
    n_reconstr = 8
    seeds = [np.random.randint(2 ** 20) for _ in range(n_reconstr)]
    maxiter = 10000
    max_chunk = None  # 2000
    # data = 'cantor'
    data = 'lena'
    # data = 'local_smooth'
    # data = 'single_freq_modulated_bis'

    compact = "smooth"

    overwrite_save_dir = 'trash_error'
    # overwrite_save_dir = None

    do_cuda = True  # if True, will check if CUDA is available et use the GPU accordingly
    find_offset_x = True  # set to False if there is no point translating x to match x0

    # facts = [0.1]
    facts = [1.]
    # facts = [1., .1, .01, .001]

    print()
    cuda = cuda_available()
    if do_cuda or not cuda:
        print("CUDA available: {}\n".format(cuda))
    else:
        print("CUDA denied\n".format(cuda))
        cuda = False

    set_to_zero = True

    nscales_l = [9]
    
    # Qs = [2]
    Qs = [1, 2]

    # nocts = [8]
    nocts = [1, 2, 3, 4, 5, 6, 7, 8]

    delta_k = [0]
    num_k_modulus = 3

    # wavelet_types = ["bump_steerable", "battle_lemarie"]
    # wavelet_types = ["battle_lemarie"]
    wavelet_types = ["bump_steerable"]
    # high_freqs = list(np.linspace(0.35, 0.5, 16))
    high_freqs = [0.425]

    exp_desc = colored('Experiments', 'yellow')
    for exp in range(n_exp):
        T = 1024
        if data == 'lena':
            # line_idx = int(np.random.randint(512))
            line_idx = 448
            x0_raw, line_idx = signal.lena_line(line_idx=line_idx, compact=compact)
            extra_info = {'line_idx': line_idx, 'border': compact}
            save_dir = data + '{}{}_pruned'.format(line_idx, compact)
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
            n_set = 10
            x0_raw = signal.locally_smooth(T, n_set, compact=compact)
            extra_info = {'n_set': n_set, 'border': compact}
            save_dir = 'locallysmooth{}{}_pruned_{}'.format(n_set, compact, int(np.random.randint(100000)))
        elif data == 'single_freq_modulated':
            per0, per1 = 11., 127.
            x0_raw = signal.single_freq_modulated(T, per0=per0, per1=per1)
            extra_info = {'per0': per0, 'per1': per1}
            save_dir = 'single_freq_modulated'
        elif data == 'single_freq_modulated_bis':
            per0, per1 = 5., 127.
            x0_raw = signal.single_freq_modulated_bis(T, per0=per0, per1=per1, compact="padd")
            extra_info = {'per0': per0, 'per1': per1, 'border': compact}
            save_dir = 'single_freq_modulated_bis_{}'.format(compact)
        elif data == 'cantor':
            a1, a2 = 1/3, 2/3  # size factors for left / right cantors
            b1, b2 = 0.5, 0.5  # weight factors for left / right cantors
            x0_raw = signal.make_cantor(T, a1, a2, b1, b2, zero_mean=False)
            extra_info = {'a1': a1, 'a2': a2, 'b1': b1, 'b2': b2}
            save_dir = 'cantor_pruned_a1{}a2{}b1{}b2{}'.format(
                np.round(a1, 2), np.round(a2, 2), np.round(b1, 2), np.round(b2, 2)
            )
        else:
            raise ValueError("Unknown data: '{}'".format(data))

        if overwrite_save_dir is not None:
            save_dir = overwrite_save_dir

        # plt.figure()
        # plt.plot(x0_raw.data.cpu().numpy()[0, 0])
        # plt.show()

        exp_desc = colored('Experiment {}/{}'.format(exp + 1, n_exp), 'yellow')
        params_order = ('J', 'Q', 'Jmax', 'wav_type', 'fact', 'high_freq')
        for num_exp, seed in enumerate(tqdm(seeds, desc=exp_desc)):
            args = list(product(nscales_l, Qs, nocts, wavelet_types, facts, high_freqs))
            init_desc = 'Parameter set'.format(num_exp + 1, len(seeds))
            with tqdm(args, desc=colored(init_desc, 'yellow'), leave=False) as t:
                for params in t:

                    nscales, Q, noct, wavelet_type, fact, high_freq = params
                    emb_descr = "Embedding: " + ' '.join(
                        ['{}={}'.format(name, val) for name, val in zip(params_order, params)])

                    # set random seed
                    tqdm.write('Random seed used : {}'.format(seed))
                    np.random.seed(seed)
                    torch.manual_seed(seed + 1)
                    torch.cuda.manual_seed(seed + 2)

                    # generate data
                    x0 = x0_raw.clone()

                    signal_info = data
                    x0 = x0.cpu().numpy()[0, 0]
                    x = fact * np.random.randn(*x0.shape) / np.sqrt(x0.size)
                    T = x0.shape[-1]

                    ndiag = noct * Q + 1

                    phi = PhaseHarmonicPruned(
                        nscales, Q, T, wav_type=wavelet_type,
                        delta_j=noct, high_freq=high_freq,
                        delta_k=delta_k, num_k_modulus=num_k_modulus,
                        check_for_nan=False, max_chunk=max_chunk)

                    # loss_fn = mse_loss
                    loss_fn = MSELoss(phi)

                    function_obj = SolverHack(phi, x0, loss_fn, cuda=cuda)
                    fst_order_dim, scd_order_dim = phi.shape()  # size of the embedding in complex numbers
                    num_coeff = phi.num_coeff()  # number of coefficients

                    check_conv_criterion = CheckConvCriterion(function_obj, 1e-24)

                    tqdm.write(colored(emb_descr, 'red'))
                    tqdm.write(colored("Using embedding " + function_obj.embedding.__class__.__name__, 'red'))
                    tqdm.write(colored("Embedding using {} coefficients.".format(num_coeff), 'red'))

                    wavelet_info = wavelet_type.replace("_", "-") + '{:.3f}'.format(high_freq)
                    save_name = signal_info + "{}coeff_{}_{}_N{}_Q{}_init{}_seed{}".format(
                        num_coeff, wavelet_info, 'MSE', nscales, Q, fact, seed)
                    tqdm.write(save_name)


                    tic = time()

                    method = 'L-BFGS-B'
                    # method = 'CG'

                    func = function_obj.joint
                    res = opt.minimize(
                        func, x, method=method, jac=True, tol=1e-16,
                        callback=check_conv_criterion, options={'maxiter': maxiter}
                    )
                    final_loss, x_opt, niter, msg = res['fun'], res['x'], res['nit'], res['message']

                    toc = time()
                    final_loss, final_grad = function_obj.joint(x_opt)
                    final_gloss = np.linalg.norm(final_grad, ord=float('inf'))
                    err_logs = check_conv_criterion.logs
                    gerr_logs = check_conv_criterion.glogs

                    tqdm.write(colored("    ----    ", 'blue'))

                    if not isinstance(msg, str):
                        msg = msg.decode("ASCII")
                    tqdm.write(colored('Optimization Exit Message : ' + msg, 'blue'))
                    tqdm.write(colored("found parameters in {}s, {} iterations -- {}it/s".format(
                        round(toc - tic, 4), niter, round(niter / (toc - tic), 2)), 'blue'))
                    tqdm.write(colored("    relative error {:.3E}".format(final_loss), 'blue'))
                    tqdm.write(colored("    relative gradient error {:.3E}".format(final_gloss), 'blue'))
                    tqdm.write(colored("    x0 norm {:.3E}".format(float(function_obj.err0.data.cpu().numpy())), 'blue'))

                    if find_offset_x:
                        offset = offset_greed_search_psnr(x_opt, x0)
                        x_opt = np.roll(x_opt, offset)

                        psnr = PSNR(x_opt, x0, 2.)  # signal values are always in [-1, 1]
                        tqdm.write(colored("PSNR : {}".format(psnr), 'green'))
                    else:
                        psnr = None

                    save_path = join(RESPATH, "gen_phaseexp_inv/", save_dir)
                    make_dir_if_not_there(save_path)
                    save_var = {
                        'x0': x0, 'x': x_opt, 'psnr': psnr, 'seed': seed,
                        'N': nscales, 'Q': Q, 'Jmax': noct, 'T': T, 'wav_type': wavelet_type,
                        'final_loss': final_loss, 'num_coeff': num_coeff,
                        'fst_order_dim': fst_order_dim, 'scd_order_dim': scd_order_dim,
                        'high_freq': high_freq, 'data_name': data,
                        'err_logs': err_logs, 'gerr_logs': gerr_logs,
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
