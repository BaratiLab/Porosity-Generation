import sys
import os
if __name__ == "__main__":
    sys.path.append("../pyscatwave")
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, grad
import torch.cuda
import librosa.core
from global_const import DATAPATH, RESPATH, Tensor
import metric
import optim


if __name__ == "__main__":
    save, is_true_test = True, True
    if not save:
        print("WARNING: experiment will not be saved")
    cuda = True
    check_for_nan = False
    detail = False

    seed = np.random.randint(10 ** 4)

    N = 12
    Q = 12
    tol = 1.5
    wavelet_type = 'battle_lemarie'

    # filenames = ["dog-bark_original.wav", "flute_original.wav", "speech_original.wav"]
    filenames = ["flute_original.wav"]
    subsample = True

    for noct in [4]:
        ndiag = noct * Q + 1
        K = max(3, noct + 1)
        for filename in filenames:

            np.random.seed(seed)
            torch.manual_seed(seed + 1)
            torch.cuda.manual_seed(seed + 2)
            print("Random seed: {}".format(seed))

            load_path = os.path.join(DATAPATH, "gen_phaseexp_inv", filename)
            print("loading from: '{}'".format(load_path))
            x0_np, sr = librosa.core.load(load_path)
            if subsample:
                print('subsampling')
                new_sr = 4096
                x0_np = librosa.core.resample(x0_np, sr, new_sr)
                sr = new_sr


            x0_np = x0_np[4096:6144]


            T = x0_np.size
            print("total time steps: {} at {}s/s".format(T, sr))
            x0 = Tensor(x0_np[None, None, :])

            phi_scd = metric.PhaseHarmonicCov(
                N, Q, K, T, ndiag=ndiag, tolerance=tol, wav_type=wavelet_type,
                fst_order=True, multi_same=False, check_for_nan=check_for_nan)
            algo_scd = optim.AdamDescentScd('MSE', detail=detail)

            print("Initialized embedding with {} parameters".format(phi_scd.shape()))

            if cuda:
                algo_scd = algo_scd.cuda()
                phi_scd = phi_scd.cuda()
                x0 = x0.cuda()

            niter = 32000
            lr0 = 0.1
            gamma = 0.85
            # milestones = []
            milestones = [1000 * i for i in range(1, 32)]
            x, logs_scd = algo_scd(x0, phi_scd, niter=niter, print_freq=1000,
                                   lr=lr0, milestones=milestones, gamma=gamma)
            print("Second Order Loss: {}".format(logs_scd[0][-1]))

            x0_np = x0.cpu().squeeze(1).squeeze(0).numpy()
            x_np = x.cpu().squeeze(1).squeeze(0).numpy()

            if save:
                save_var = {
                    'x0': x0_np, 'x': x_np, 'seed': seed, 'sr': sr,

                    'N': N, 'Q': Q, 'K': K, 'ndiag': ndiag, 'T': T, 'tol': phi_scd.tol,
                    'fst_order': phi_scd.fst_order, 'wav_type': wavelet_type,
                    'multi_same': phi_scd.multi_same,
                    
                    'logs': np.array(logs_scd[0]),
                    'logs_fst': np.stack(logs_scd[1], axis=-1) if detail else None,
                    'logs_scd': np.stack(logs_scd[2], axis=-1) if detail else None,
                }

                data_save_path = os.path.join(RESPATH, 'gen_phaseexp_inv', 'sound')
                name = filename.split('_')[0]
                save_name = name + '{}_'.format(sr) + phi_scd.__class__.__name__ + '_' + \
                    'N{}Q{}K{}_{}diag_'.format(N, Q, K, ndiag) + algo_scd.__class__.__name__ + '_' + str(seed)
                if not is_true_test:
                    save_name = 'debug_' + save_name
                x_np = x.cpu().numpy()
                np.savez(os.path.join(data_save_path, save_name), **save_var)

