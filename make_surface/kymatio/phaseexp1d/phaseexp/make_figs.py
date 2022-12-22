import sys
import os
if __name__ == "__main__":
    sys.path.append("../pyscatwave")
from itertools import product
import math
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda

from global_const import *
from torch.autograd import Variable, grad
try:
    from scattering.scattering1d.utils import modulus
    from scattering.scattering1d import filter_bank
    from scattering.scattering1d.fft_wrapper import fft1d_c2c, ifft1d_c2c_normed
except ImportError:
    from scatwave.scattering1d.utils import modulus
    from scatwave.scattering1d import filter_bank
    from scatwave.scattering1d.fft_wrapper import fft1d_c2c, ifft1d_c2c_normed

from tqdm import tqdm

from global_const import DATAPATH, Tensor
import metric
import optim
from utils import make_dir_if_not_there


def figure(x=1, y=1):
    a = max(x, y)
    r = 12 / a
    return plt.figure(figsize=(x * r, y * r))


def smooth_signal(x):
    x_hat = np.fft.fft(x)
    
    sigma_hat = 0.15
    T = x.shape[-1]
    freqs = np.fft.fftfreq(T)
    gauss_hat = np.exp(- 0.5 * freqs ** 2 / sigma_hat ** 2)
    gauss_hat = gauss_hat / (np.sqrt(np.mean(gauss_hat ** 2, axis=-1) * 2 * np.pi))

    x_smooth = np.real(np.fft.ifft(x_hat * gauss_hat))

    # figure(x=2)
    # plt.plot(freqs, gauss_hat)
    # plt.axhline(y=0)
    # plt.show()
    
    # figure(x=2)
    # plt.plot(x, 'r')
    # plt.plot(x_smooth, 'b')
    # plt.show()
    
    # figure(x=2)
    # plt.plot(np.fft.fftshift(freqs), np.fft.fftshift(np.abs(x_hat)), 'r')
    # plt.semilogy(np.fft.fftshift(freqs), np.fft.fftshift(np.abs(np.fft.fft(x_smooth))), 'b')
    # plt.show()

    # raise SystemExit

    return x_smooth

def solve_border(x0, opt):
    T = np.size(x0)
    if opt is None:
        pass
    elif opt == "padd":
        zeros = np.zeros(T // 2)
        x0 = np.concatenate((zeros, x0, zeros))
    elif opt == "smooth":
        sin2 = np.sin(np.linspace(0, np.pi / 2, T // 8)) ** 2
        enveloppe = np.concatenate((sin2, np.ones(3 * T // 4), sin2[::-1]))
        x0 = x0 * enveloppe
    elif opt == "mirror":
        x0 = np.concatenate((x0, x0[::-1]))
    else:
        err = "opt should be one of [None, 'padd', 'smooth', 'mirror'], but got '{}'".format(opt)
        raise ValueError(err)
    return x0

def diracs(T, n_dirac):
    x0 = torch.zeros(1, 1, T)
    loc = np.random.choice(T // 2, size=n_dirac, replace=False)
    ampl = np.random.randn(n_dirac)
    for l, a in zip(loc, ampl):
        x0[0, 0, l + T // 4] = a
    return x0

def staircase(T, n_dirac, compact=True, zero_mean=True, smooth=True):
    if compact:
        t = T // 2
    else:
        t = T
    x0 = diracs(t, n_dirac)
    x0 = x0.numpy()[0, 0, :]
    x0 = np.cumsum(x0, axis=0)
    if zero_mean:
        x0 = x0 - np.mean(x0)
    if compact:
        zeros = np.zeros(T // 4)
        x0 = np.concatenate((zeros, x0, zeros), axis=0)
    if smooth:
        x0 = smooth_signal(x0)
    return Tensor(x0[None, None, :])

def locally_smooth(T, n_ensemble, compact=True, zero_mean=False, smooth=True, per=2.):
    if compact == "padd" or compact == "mirror":
        t = T // 2
    else:
        t = T
    x0 = np.zeros(t)
    loc_discontinuity = np.sort(np.random.choice(t, size=2 * n_ensemble, replace=False))
    loc_discontinuity = loc_discontinuity
    ampl = np.random.randn(n_ensemble)
    for k, a in enumerate(ampl):
        i, j = loc_discontinuity[2 * k], loc_discontinuity[2 * k + 1]
        x0[i:j] = a
    var = np.cos(np.linspace(0, 2 * np.pi, t, endpoint=False) * per)
    x0 = x0 * var

    if zero_mean:
        x0 = x0 - np.mean(x0)
    x0 = solve_border(x0, compact)
    if smooth:
        x0 = smooth_signal(x0)
    return Tensor(x0[None, None, :])

def single_freq_modulated(T, per0=11., per1=127., compact=True, zero_mean=False, smooth=True):
    if compact:
        t = T // 2
    else:
        t = T
    t = np.linspace(0, 2 * np.pi, t)
    theta = np.random.rand() * 2 * np.pi
    x0 = np.sin(t * per0) * np.cos(t * per1 + theta)

    if zero_mean:
        x0 = x0 - np.mean(x0)
    if compact:
        zeros = np.zeros(T // 4)
        x0 = np.concatenate((zeros, x0, zeros), axis=0)
    if smooth:
        x0 = smooth_signal(x0)
    return Tensor(x0[None, None, :])

def single_freq_modulated_bis(T, per0=5., per1=127., compact=True, zero_mean=False, smooth=True):
    if compact:
        t = T // 2
    else:
        t = T
    t = np.linspace(0, 2 * np.pi, t)
    theta = np.random.rand() * 2 * np.pi
    x0 = -(np.cos(t * per0) - 1.) * np.cos(t * per1 + theta)

    if zero_mean:
        x0 = x0 - np.mean(x0)
    x0 = solve_border(x0, compact)
    if smooth:
        x0 = smooth_signal(x0)
    return Tensor(x0[None, None, :])


def fourier_diracs(T, n_dirac):
    x0_hat_re = np.zeros(T)
    x0_hat_im = np.zeros(T)
    loc = np.random.choice(T // 2 + 1, size=n_dirac, replace=False)
    ampl = np.random.randn(n_dirac)
    iampl = np.random.randn(n_dirac)
    for l, a, b in zip(loc, ampl, iampl):
        x0_hat_re[l] = a
        x0_hat_im[l] = b
    x0_hat = x0_hat_re + 1j * x0_hat_im
    x0_hat_shift = np.fft.fftshift(x0_hat)
    x0_hat_negfreq = np.fft.ifftshift(np.concatenate((x0_hat_shift[0:1], x0_hat_shift[:0:-1])))
    x0_hat = (x0_hat + np.conj(x0_hat_negfreq)) / 2
    x0_np = np.fft.ifft(x0_hat)
    x0 = Tensor(np.real(x0_np)[None, None, :])
    return x0

def fourier_staircase(T, n_dirac):
    x0_hat_re = np.zeros(T)
    x0_hat_im = np.zeros(T)
    loc = np.random.choice(T // 2 + 1, size=n_dirac, replace=False)
    ampl = np.random.randn(n_dirac)
    iampl = np.random.randn(n_dirac)
    for l, a, b in zip(loc, ampl, iampl):
        x0_hat_re[l] = a
        x0_hat_im[l] = b
    x0_hat = x0_hat_re + 1j * x0_hat_im
    x0_hat = np.cumsum(x0_hat)
    x0_hat_shift = np.fft.fftshift(x0_hat)
    x0_hat_negfreq = np.fft.ifftshift(np.concatenate((x0_hat_shift[0:1], x0_hat_shift[:0:-1])))
    x0_hat = (x0_hat + np.conj(x0_hat_negfreq)) / 2
    x0_np = np.fft.ifft(x0_hat)
    x0 = Tensor(np.real(x0_np)[None, None, :])
    return x0

def lena_line(line_idx=None, compact="padd", zero_mean=True, smooth=True):
    # lena = plt.imread(os.path.join(DATAPATH, "gen_phaseexp_inv/lena512c.jpg"))
    lena_rgb = plt.imread(os.path.join(DATAPATH, "gen_phaseexp_inv/lena1024c.jpg"))
    lena = np.dot(lena_rgb[..., :3], [0.299, 0.587, 0.114])
    T = lena.shape[-1]

    if line_idx is None:
        line_idx = np.random.randint(T // 4, 3 * T // 4)

    x0 = np.array(lena[line_idx, :])
    if zero_mean:
        x0 = x0 - np.mean(x0)
    x0 = solve_border(x0, compact)
    if smooth:
        x0 = smooth_signal(x0)

    x0 = Tensor(x0[None, None, :])
    x0 = x0 / torch.max(torch.abs(x0))

    return x0, line_idx

def make_cantor(n, a1, a2, b1, b2, compact=True, zero_mean=True, smooth=True):
    """Recursively generates a Cantor distribution.
    Inputs:
        n: size of output
        a1: size factor for left cantor
        a2: size factor for right cantor
        b1: weight factor for left cantor
        b2: weight factor for right cantor
    """

    def make_cantor_rec(x, n1, n2, p):
        if n2 < n1:
            raise ValueError("Left or right border misplaced.")
        elif n2 - n1 <= 6:
            x[int(n1):int(n2)] = p / (int(n2) - int(n1))
        else:
            make_cantor_rec(x, n1, n1 + (n2 - n1) * a1, p * b1)
            make_cantor_rec(x, n1 + (n2 - n1) * a2, n2, p * b2)

    if compact:
        n_used = n // 2
    else:
        n_used = n

    x0 = np.zeros(n_used)
    make_cantor_rec(x0, 0, n_used, 1.)

    if zero_mean:
        x0 = x0 - np.mean(x0)
    if compact:
        zero = np.zeros(n // 4)
        x0 = np.concatenate((zero, x0, zero), axis=0)
    if smooth:
        x0 = smooth_signal(x0)

    x0 = Tensor(x0[None, None, :])
    x0 = x0 / torch.max(torch.abs(x0))

    return x0


def binomial_cascade(p, **kwargs):
    b1, b2 = p, 1 - p
    return make_cantor(0.5, 0.5, b1, b2, **kwargs)



def random_notes(n, n_freq, n_harm, time_lim, sr):
    tmin, tmax = time_lim

    min_freq = 10 * sr / tmin  # at least 10 periods at the same frequency
    max_freq = sr / (2 * n_harm)  # all harmonies must be within the signal's quality
    
    n_oct = np.log2(max_freq / min_freq)
    n_note = int(n_oct * 12)  # number of possible fundamental frequencies
    assert(max_freq > min_freq)

    fund_idxs = np.random.choice(n_note, size=n_freq, replace=False)
    fund_freqs = min_freq * np.power(2, fund_idxs / 12)

    n_notes = int(n / tmax)
    length_notes = np.random.randint(tmin, tmax, size=n_notes)
    idx_notes = np.random.randint(n_freq, size=n_notes)

    x0 = np.zeros((n,))
    harm = np.arange(1, n_harm + 1)[None]
    ampl = np.power(2., 1 - harm)
    for i, (length, idx_freq) in enumerate(zip(length_notes, idx_notes)):
        # get time range
        idx_start = i * tmax
        idx_stop = idx_start + length

        # get amplitude decrease over time
        increase = np.cos(np.linspace(-np.pi / 2, 0, length // 12)) ** 2
        plateau = np.ones(length // 2 - length // 12)
        decrease = np.cos(np.linspace(0, np.pi / 2, length - length // 2)) ** 2
        envelope = np.concatenate((increase, plateau, decrease))

        # find fundamental frequency and its harmonies
        fund_freq = fund_freqs[idx_freq]
        freqs = fund_freq * harm

        # make oscillation
        t = np.arange(0, length)[:, None]
        osc = np.sum(np.sin(2 * np.pi * freqs * t / sr) * ampl, axis=-1)

        # multiply by envelope and write on signal
        x0[idx_start:idx_stop] = osc * envelope

    x0 = Tensor(x0[None, None, :])
    x0 = x0 / torch.max(torch.abs(x0))

    return x0


def offset(x, x0, phi, order=2):
    wav = phi.filt_hat[order, ...].data.cpu().numpy()
    wav = wav[..., 0] + 1j * wav[..., 1]
    
    x0_hat = np.fft.fft(x0)
    x_hat = np.fft.fft(x)
    x0_filt_hat = x0_hat * wav
    x_filt_hat = x_hat * wav
    x0_filt = np.fft.ifft(x0_filt_hat)
    x_filt = np.fft.ifft(x_filt_hat)
    
    T = np.size(wav)
    t = np.arange(T)
    x0_center = np.sum(t * np.abs(x0_filt) ** 2) / np.sum(np.abs(x0_filt) ** 2)
    x_center = np.sum(t * np.abs(x_filt) ** 2) / np.sum(np.abs(x_filt) ** 2)

    return int(round(x0_center - x_center))


def offset_greed_search(x, x0, order=2):
    T = np.size(x)
    min_err, min_offset = float('inf'), -1
    for offset in range(T):
        x1 = np.roll(x, offset)
        err = np.linalg.norm(x0 - x1, ord=order)
        if err < min_err:
            min_err, min_offset = err, offset

    return min_offset


if __name__ == "__main__":

    nb_exp = 2  # number of iterations
    seed = [np.random.randint(10 ** 6) for _ in range(nb_exp)]
    seed = [411602]
    # seed = [354934]  # use for graphics

    wavelet_types = ["battle_lemarie"]  # ["battle_lemarie", "morlet"]
    do_fst_order = [True]  # [True, False]
    n_octave = list((1, 6))  # number of octaves of interactions kept in covariance matrix
    loss_types = ['MSE']  # ['RelativeMSE', 'MSE', 'L1]
    cuda = True
    zero_means = [True]  # 
    smooths = [False]
    nscales_list = [12]
    Qs = list((1, 8))
    # Qs = [2]
    high_freq_bl = 0.5
    Ks = [1 + 2 ** k for k in range(1, 6)]
    # save_dir = "order1_usefulness"
    # save_dir_root = "lena/error_curve_2"
    save_dir_root = "graphics"
    #signal_name = "lena"
    signal_name = "cantor"
    #signal_name = "staircase"
    detail = True

    
    Ts = [2 ** 10]
    n_diracs_l = [4]

    for rs in seed:
        for wavelet_type, fst_order, nscales,K,  Q, noct, loss_type, zero_mean, smooth, T, n_diracs \
            in product(wavelet_types, do_fst_order, nscales_list, Ks, Qs, n_octave, \
                       loss_types, zero_means, smooths, Ts, n_diracs_l):
        # for wavelet_type, fst_order, K, ndiag, loss_type, zero_mean, smooth, T, n_diracs \
        #     in product(wavelet_types, do_fst_order, Ks, do_reduce, loss_types, \
        #                zero_means, smooths, Ts, n_diracs_l):

            # manual seed
            np.random.seed(rs)
            torch.manual_seed(rs + 1)
            torch.cuda.manual_seed(rs + 2)
            print("Random seed: {}".format(rs))

            # some hyperparameters deduced from others
            ndiag = int(Q * noct + 1)
            # K = max(3, noct + 1)
            # K = 3
            save_dir = save_dir_root
            # save_dir = save_dir_root + '_' + str(T)

            save_path = join(RESPATH, join("gen_phaseexp_inv/", save_dir))
            make_dir_if_not_there(save_path)

            # choose signal to be reconstructed
            if signal_name == 'lena':
                signal, line_idx = lena_line(zero_mean=zero_mean)
                signal_info = 'lena' + str(line_idx)
            elif signal_name == 'cantor':
                signal = make_cantor(T, 0.3333, 0.6666, 0.5, 0.5, zero_mean=zero_mean, smooth=smooth)
                signal_info = 'cantor'
            elif signal_name == 'staircase':
                signal = staircase(T, n_diracs, zero_mean=zero_mean, smooth=smooth)
                signal_info = 'staircase'
            else:
                raise ValueError("Unknown signal name: {}".format(signal_name))
            x0 = signal
            T = x0.size(-1)

            # initialize metric
            tol = 2.0
            # phi_fst = metric.PhaseHarmonicAvg(
            #     nscales, Q, K, T, wav_type=wavelet_type, check_for_nan=False)
            phi_scd = metric.PhaseHarmonicCov(
                nscales, Q, K, T, wav_type=wavelet_type, ndiag=ndiag,
                fst_order=fst_order, tolerance=tol, multi_same=False,
                high_freq=high_freq_bl, check_for_nan=False)
            reduced = "reduced" + str(noct) + "oct"

            # initialize and print save name
            high_freq_info = (str(high_freq_bl) if wavelet_type == "battle_lemarie" else "")
            wavelet_info = wavelet_type.replace("_", "-") + high_freq_info
            if zero_mean:
                signal_info = signal_info + "-0mean"
            if smooth:
                signal_info = signal_info + "-smooth"
            save_name = signal_info + "_{}o2_{}-{}coeff_{}_{}_N{}_Q{}_K{}_seed{}".format(
                "o1" if fst_order else "", reduced, phi_scd.shape_effect(),
                wavelet_info, loss_type, phi_scd.N, phi_scd.Q, phi_scd.K, rs)
            print(save_name)

            # initialize solvers
            # algo_fst = optim.AdamDescentFst(loss_type, detail=detail)
            algo_scd = optim.AdamDescentScd(loss_type, detail=detail)

            # move to GPU
            if cuda:
                # algo_fst = algo_fst.cuda()
                algo_scd = algo_scd.cuda()
                # phi_fst = phi_fst.cuda()
                phi_scd = phi_scd.cuda()
                x0 = x0.cuda()
                print("Working on GPU")
            else:
                # algo_fst = algo_scd.cpu()
                algo_fst = algo_scd.cpu()
                # phi_fst = phi_fst.cpu()
                phi_scd = phi_scd.cpu()
                x0 = x0.cpu()
                print("Working on CPU")

            x = None

            # First, optimize first order coefficients
            # x, logs_fst = algo_fst(x0, phi_fst, niter=8000, print_freq=1000, lr=0.01)
            # stops_fst = [len(logs_fst[0])]
            # print("First Order Loss: {}".format(logs_fst[0][-1]))

            # Second, optimize covariance matrix
            niter = 32000
            lr0 = 0.1
            gamma = 0.9
            # milestones = []
            milestones = [1000 * i for i in range(1, 32)]
            x, logs_scd = algo_scd(x0, phi_scd, niter=niter, print_freq=1000, past_x=x,
                                   lr=lr0, milestones=milestones, gamma=gamma)
            print("Second Order Loss: {}".format(logs_scd[0][-1]))

            # compute final loss
            final_loss = logs_scd[0][-1]
            save_name = save_name + "_loss{:.1E}".format(final_loss)

            # convert to numpy
            x0_np = x0.cpu().squeeze(1).squeeze(0).numpy()
            x_np = x.cpu().squeeze(1).squeeze(0).numpy()

            # quick align
            t = np.arange(T)
            # off = offset(x_np, x0_np, phi_scd, order=Q + 1)
            off = offset_greed_search(x_np, x0_np, order=2)
            x_np_centered = np.roll(x_np, off)
            err = np.linalg.norm(x0_np - x_np, ord=2) / np.linalg.norm(x0_np, ord=2)

            # # plot in time
            # fig = plt.figure(figsize=(12, 12))
            # ax = plt.subplot2grid((4, 1), (0, 0), rowspan=3)
            # ax.plot(x0_np, 'r')
            # ax.plot(x_np_centered, 'b')
            # ax.set_xticklabels([])
            # plt.title("Original (red) and reconstructed (blue) signals")
            # ax = plt.subplot2grid((4, 1), (3, 0), rowspan=1)
            # ax.plot(x0_np - x_np_centered, 'r')
            # plt.title("Error")
            # plt.savefig(join(save_path, save_name + "_time.pdf"))

            # # plot in fourier
            # fig = plt.subplot(2, 1, 2)
            # ax = plt.subplot2grid((5, 1), (0, 0), rowspan=3)
            # ax.semilogy(np.abs(np.fft.fftshift(np.fft.fft(x0_np))), 'r')
            # ax.semilogy(np.abs(np.fft.fftshift(np.fft.fft(x_np))), 'b')
            # ax.set_xticklabels([])
            # plt.title("Original (red) and reconstructed (blue) Fourier spectra")
            # ax = plt.subplot2grid((5, 1), (3, 0), rowspan=1)
            # ax.semilogy(np.abs(np.fft.fftshift(np.fft.fft(x0_np) - np.fft.fft(x_np))), 'r')
            # ax.set_xticklabels([])
            # plt.title("Fourier Error")
            # ax = plt.subplot2grid((5, 1), (4, 0), rowspan=1)
            # ax.semilogy(np.abs(np.fft.fftshift(np.fft.fft(x0_np) - np.fft.fft(x_np))) / np.abs(np.fft.fftshift(np.fft.fft(x0_np))), 'r')
            # plt.title("Fourier Relative Error")
            # plt.savefig(join(save_path, save_name + "_fourier.pdf"))

            # save experiment
            if detail:
                logs1 = np.stack(logs_scd[1], axis=-1)
                logs2 = np.stack(logs_scd[2], axis=-1)
            else:
                logs1 = None
                logs2 = None
            save_var = {
                'x0': x0_np, 'x': x_np, 'err': err, 'order': 2, 'seed': rs,

                'N': nscales, 'Q': Q, 'K': K, 'ndiag': ndiag, 'T': T, 'tol': phi_scd.tol,
                'fst_order': phi_scd.fst_order, 'wav_type': wavelet_type,
                'multi_same': phi_scd.multi_same,
                
                # 'logs_prepare': np.array(logs_fst[0]),
                # 'logs_prepare_detail': np.stack(logs_fst[1], axis=-1) if detail else None,

                'logs': np.array(logs_scd[0]),
                'logs_fst': logs1,
                'logs_scd': logs2,
            }
            np.savez(join(save_path, save_name + "_x.npz"), **save_var)
            # fig_loss.savefig(join(save_path, save_name + "_loss.pdf"))
            print('saved as "{}_[signal.pdf, loss.pdf, x.npz]"'.format(join(save_path, save_name)))
            print(save_name + "_time")
            print("\n--------------------------------------------\n")
        