import sys
if __name__ == "__main__":
    sys.path.append("../pyscatwave/")
from os.path import join
from math import sqrt
import numpy as np
import scipy as sp
import scipy.optimize as opt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, grad

from time import time
from tqdm import tqdm
from termcolor import colored

from utils import cuda_available, make_dir_if_not_there
from itertools import product
from global_const import RESPATH, DATAPATH, Tensor

import complex_utils as cplx
import make_figs as signal
from loss import PSNR
from full_embedding import FullEmbedding
from check_conv_criterion import CheckConvCriterion, SmallEnoughException
import librosa.core



class SolverHack(nn.Module):
    def __init__(self, embedding, x0, loss_fn, cuda=False):
        super(SolverHack, self).__init__()

        self.embedding = embedding
        self.x0 = x0
        self.loss = loss_fn
        self.is_cuda = False
        self.res = None, None
        self.loss_logs = None, None

        if cuda:
            self.cuda()

        # compute embedding and loss at initial guess
        x0_torch = self.format(self.x0, requires_grad=False)
        self.emb0 = self.embedding(x0_torch)
        self.loss_scat0, self.loss_harm0 = self.loss(self.emb0, None)
        self.err0 = self.loss.combine(self.loss_scat0, self.loss_harm0)

    def cuda(self):
        super(SolverHack, self).cuda()
        if not self.is_cuda:
            self.is_cuda = True
            self.embedding = self.embedding.cuda()
            if 'emb0' in self.__dict__:
                self.emb0 = self.emb0.cuda()
        return self

    def cpu(self):
        super(SolverHack, self).cpu()
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
        loss_scat, loss_harm = self.loss(emb, self.emb0)
        if self.loss_scat0 > 0:
            loss_scat /= self.loss_scat0
        if self.loss_harm0 > 0:
            loss_harm /= self.loss_harm0
        loss = self.loss.combine(loss_scat, loss_harm)

        # compute gradient
        grad_x, = grad([loss], [x_torch], retain_graph=True)

        # only get the real part
        grad_x = grad_x[0, 0, ..., 0]

        # move to numpy
        grad_x = grad_x.contiguous().detach().data.cpu().numpy()
        loss = loss.detach().data.cpu().numpy()

        self.res = loss, grad_x
        loss_scat = loss_scat.detach().cpu().numpy()
        loss_harm = loss_harm.detach().cpu().numpy()
        self.loss_logs = loss_scat, loss_harm

        return loss, grad_x


class MSELoss(nn.Module):
    def __init__(self, phi, alpha=0.5, use_cuda=False):
        super(MSELoss, self).__init__()
        self.phi = phi
        self.alpha = alpha
        print(self.alpha)

        self.is_cuda_p = use_cuda

        self.loss_scat = None
        self.loss_harm = None

    def forward(self, input, target):

        i_f, i_s = input
        i_f_scat = i_f[0]
        i_f_phe  = i_f[1]
        i_s_scat = i_s[0]
        i_s_phe  = i_s[1]

        if target is None:
            t_f_scat = torch.zeros_like(i_f_scat)
            t_f_phe  = torch.zeros_like(i_f_phe)
            t_s_scat = torch.zeros_like(i_s_scat)
            t_s_phe  = torch.zeros_like(i_s_phe)
        else:
            t_f, t_s = target
            t_f_scat = t_f[0]
            t_f_phe  = t_f[1]
            t_s_scat = t_s[0]
            t_s_phe  = t_s[1]

        gap_scat = self.compute_gap_scat(i_f_scat, i_s_scat, t_f_scat, t_s_scat).double()
        gap_phe = self.compute_gap_phe(i_f_phe, i_s_phe, t_f_phe, t_s_phe)


        # Compute and return mse loss on gaps
        loss_scat = self.mse_norm(gap_scat)
        loss_harm = self.mse_norm(gap_phe)

        return loss_scat, loss_harm


    def combine(self, loss_scat, loss_harm):
        loss = self.alpha * loss_scat + (1 - self.alpha) * loss_harm
        return loss

    def compute_gap_scat(self, i_f, i_s, t_f, t_s):
        if i_s.shape == torch.Size([0]):
            gap = Tensor([0])
            if self.is_cuda_p:
                gap = gap.cuda()
            return gap

        s_gap = t_s - i_s
        f_gap = t_f[...,1] - i_f[...,1] - 2*t_f[...,0]*(t_f[...,0] - i_f[...,0])
        gap = torch.cat((f_gap, s_gap), dim=2)
        return gap

    def compute_gap_phe(self, i_f, i_s, t_f, t_s):
        if i_s.shape == torch.Size([0]):
            gap = Tensor([0])
            if self.is_cuda_p:
                gap = gap.cuda()
            return gap
        s_gap = i_s - t_s
        f_gap = i_f - t_f

        sel = torch.index_select

        gap = []
        start = 0
        # Loop all types of selected coefficients
        for xi_idx, ks in zip(self.phi.phase_harmonic.xi_idx,
                              self.phi.phase_harmonic.ks):
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
            err_fst1 = cplx.mul(t_f0, cplx.conjugate(-f_gap1))

            l = xi_idx.size(0)
            s_gap_l = s_gap[:, :, start:start + l]
            start += l

            g = s_gap_l - err_fst0 - err_fst1
            gap.append(g)
        gap = torch.cat(gap, dim=2)
        return gap

    @staticmethod
    def mse_norm(x):
        sq_err = (x ** 2).sum(-1)
        return torch.mean(sq_err.view(-1))

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

    n_exp = 1
    n_reconstr = 4
    seeds = [np.random.randint(2 ** 20) for _ in range(n_reconstr)]
    maxiter = 50000
    # data = 'cantor'
    data = 'data'
    # data = 'lena'
    # data = 'local_smooth'
    # data = 'single_freq_modulated_bis'
    # data = 'bird'
    max_chunk = None

    compact = "smooth"

    do_cuda = True  # if True, will check if CUDA is available et use the GPU accordingly
    find_offset_x = False  # set to False if there is no point translating x to match x0

    alpha = 1e-2  # weight of scattering in loss, phase harmonic has weight 1 - alpha
    # alpha = 0.5  # weight of scattering in loss, phase harmonic has weight 1 - alpha
    facts = [1.]

    # overwrite_save_dir = 'test_sound_{:.2E}'.format(alpha)
    overwrite_save_dir = None

    print()
    cuda = cuda_available()
    if do_cuda or not cuda:
        print("CUDA available: {}\n".format(cuda))
    else:
        print("CUDA denied\n".format(cuda))
        cuda = False

    set_to_zero = True

    Js = [(12, 5)]
    Qs = [12]

    scatt_orders_l = [(1, 2)]
    # phe_coeffs_l = [('harmonic', 'mixed')]
    phe_coeffs_l = [('harmonic',)]

    # deltajs = [1, 2, 3, 4, 5, 6, 7, 8]
    deltajs = [3]
    deltaks = [(0,)]
    num_k_modulus = 0
    delta_cooc = 2

    # wavelet_types = ["bump_steerable", "battle_lemarie"]
    # wavelet_types = ["battle_lemarie"]
    wavelet_types = ["morlet"]
    # high_freqs = list(np.linspace(0.35, 0.5, 16))
    high_freqs = [0.425]

    tol = 1e-24


    exp_desc = colored('Experiments', 'yellow')
    T = 2 ** 13
    for exp in range(n_exp):
        if data == 'lena':
            # line_idx = int(np.random.randint(512))
            line_idx = 448
            x0_raw, line_idx = signal.lena_line(line_idx=line_idx, compact=compact)
            extra_info = {'line_idx': line_idx, 'border': compact}
            save_dir = data + '{}{}'.format(line_idx, compact)
        elif data == 'data':
            # filename = 'applause_2.0s_8192.wav'
            # filename = 'flute_2.0s_8192.wav'
            # filename = 'speech_4.0s_4096.wav'
            filename = 'speech_2.87s_4096.wav'
            load_path = join(DATAPATH, "gen_phaseexp_inv", filename)
            x0_raw, _ = librosa.core.load(load_path)
            rate = int(filename.split('.')[-2].split('_')[-1])
            extra_info = {'sr': rate}
            print(colored("\nSignal info: size {} at rate {}".format(x0_raw.shape, rate), 'red'))
            x0_raw = torch.Tensor(x0_raw[None, None, :])
            save_dir = '.'.join(filename.split('.')[:-1])
        elif data == 'bird':
            filename = 'XC438168-BIRD_16384.wav'
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
            save_dir = 'locallysmooth{}{}_{}'.format(n_set, compact, int(np.random.randint(100000)))
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
            save_dir = 'cantor_a1{}a2{}b1{}b2{}'.format(
                np.round(a1, 2), np.round(a2, 2), np.round(b1, 2), np.round(b2, 2)
            )
        elif data == 'staircase':
            n_dirac = 10  # number of discontinuities
            x0_raw = signal.staircase(T, n_dirac)
            extra_info = {'n_dirac': n_dirac}
            save_dir = 'staircase{}'.format(n_dirac)
        elif data == 'toy_music':
            n_freq = 3  # number of notes
            n_harm = 2  # number of harmonics
            time_lim = (256 + 128, 512)
            sr = 2048
            x0_raw = signal.random_notes(T, n_freq, n_harm, time_lim, sr)
            extra_info = {'n_freq': n_freq, 'n_harm': n_harm, 'time_lim': time_lim, 'sr': sr}
            save_dir = 'toy_music'
        else:
            raise ValueError("Unknown data: '{}'".format(data))

        if overwrite_save_dir is not None:
            save_dir = overwrite_save_dir

        # plt.figure()
        # plt.plot(x0_raw.data.cpu().numpy()[0, 0])
        # plt.show()

        exp_desc = colored('Experiment {}/{}'.format(exp + 1, n_exp), 'yellow')
        params_order = ('J', 'Q', 'deltaJ', 'deltaK', 'scatt_orders', 'phe_coeffs',
                        'wav_type', 'fact', 'high_freq')
        for num_exp, seed in enumerate(tqdm(seeds, desc=exp_desc)):
            args = list(product(
                Js, Qs, deltajs, deltaks, scatt_orders_l, phe_coeffs_l,
                wavelet_types, facts, high_freqs
                ))
            init_desc = 'Parameter set'.format(num_exp + 1, len(seeds))
            with tqdm(args, desc=colored(init_desc, 'yellow'), leave=False) as t:
                for params in t:

                    J, Q, deltaj, deltak, scatt_orders, phe_coeffs, \
                        wavelet_type, fact, high_freq = params

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


                    phe_params = {
                        'delta_j': deltaj, 'delta_k': deltak,
                        'wav_type':wavelet_type, 'high_freq':high_freq,
                        'delta_cooc': delta_cooc, 'max_chunk': max_chunk
                        }
                    scatt_params = dict()

                    if not isinstance(J, int):
                        scatt_params['J'], phe_params['J'] = J

                    phi = FullEmbedding(
                        T, J, Q, phe_params=phe_params, scatt_params=scatt_params,
                        scatt_orders=scatt_orders, phe_coeffs=phe_coeffs)
                    num_coeff, nscat = phi.shape(), phi.count_scatt_coeffs()
                    nharm = num_coeff - nscat

                    loss_fn = MSELoss(phi, alpha=alpha, use_cuda=cuda)

                    solver_fn = SolverHack(phi, x0, loss_fn, cuda=cuda)
                    if cuda:
                        phi=phi.cuda()
                        loss_fn = loss_fn.cuda()
                        solver_fn = solver_fn.cuda()

                    xini = np.random.randn(*x0[None, None].shape)
                    check_conv_criterion = CheckConvCriterion(solver_fn, 1e-24)

                    tqdm.write(colored(emb_descr, 'red'))
                    tqdm.write(colored("Using embedding " + solver_fn.embedding.__class__.__name__, 'red'))
                    tqdm.write(colored(
                        "Embedding using {} coefficients : {} scattering and {} phase harmonics".format(
                            num_coeff, nscat, nharm), 'red'))

                    wavelet_info = wavelet_type.replace("_", "-") + '{:.3f}'.format(high_freq)
                    save_name = signal_info + "{}coeff_{}_{}_N{}_Q{}_init{}_seed{}".format(
                        num_coeff, wavelet_info, 'MSE', J, Q, fact, seed)
                    tqdm.write(save_name)


                    jac = True
                    func = solver_fn.joint if jac else solver_fn.function
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


                    final_loss, final_grad = solver_fn.joint(x)
                    final_gloss = np.linalg.norm(final_grad, ord=float('inf'))

                    logs_loss = check_conv_criterion.logs_loss
                    logs_grad = check_conv_criterion.logs_grad
                    logs_scat = check_conv_criterion.logs_scat
                    logs_harm = check_conv_criterion.logs_harm

                    tqdm.write(colored("    ----    ", 'blue'))

                    if not isinstance(msg, str):
                        msg = msg.decode("ASCII")
                    tqdm.write(colored('Optimization Exit Message : ' + msg, 'blue'))
                    tqdm.write(colored("found parameters in {}s, {} iterations -- {}it/s".format(
                        round(toc - tic, 4), niter, round(niter / (toc - tic), 2)), 'blue'))
                    tqdm.write(colored("    relative error {:.3E}".format(final_loss), 'blue'))
                    tqdm.write(colored("    relative gradient error {:.3E}".format(final_gloss), 'blue'))
                    x0_norm_msg = "    x0 norm  S{:.2E}  H{:.2E}".format(
                        float(solver_fn.loss_scat0.data.cpu().numpy()),
                        float(solver_fn.loss_scat0.data.cpu().numpy())
                        )
                    tqdm.write(colored(x0_norm_msg, 'blue'))


                    if find_offset_x:
                        offset = offset_greed_search_psnr(x, x0)
                        x = np.roll(x, offset)

                        psnr = PSNR(x, x0, 2.)  # signal values are always in [-1, 1]
                        tqdm.write(colored("PSNR : {}".format(psnr), 'green'))
                    else:
                        psnr = None


                    save_path = join(RESPATH, "gen_full_inv/", save_dir)
                    make_dir_if_not_there(save_path)
                    save_var = {
                        'x0': x0, 'x': x, 'psnr': psnr, 'seed': seed,
                        'J': J, 'Q': Q, 'deltaJ': deltaj, 'deltaK': deltak,
                        'T': T, 'wav_type': wavelet_type, 'alpha': alpha,
                        'final_loss': final_loss, 'num_coeff': num_coeff,
                        'num_scat_coeff': nscat, 'num_harm_coeff': nharm,
                        'high_freq': high_freq, 'data_name': data,
                        'logs_loss': logs_loss, 'logs_grad': logs_grad,
                        'logs_scat': logs_scat, 'logs_harm': logs_harm,
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
