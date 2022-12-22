import sys
if __name__ == "__main__":
    sys.path.append("../pyscatwave")
import math
from itertools import product
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, grad
import complex_utils as cplx
try:
    from scattering.scattering1d.utils import modulus
    from scattering.scattering1d import filter_bank as fb
    from scattering.scattering1d.fft_wrapper import fft1d_c2c, ifft1d_c2c_normed
except ImportError:
    from scatwave.scattering1d.utils import modulus
    from scatwave.scattering1d import filter_bank as fb
    from scatwave.scattering1d.fft_wrapper import fft1d_c2c, ifft1d_c2c_normed
from phaseexp import PhaseExp, PhaseHarmonic
import filter_bank as lfb
from utils import HookDetectNan, count_nans
from global_const import Tensor

class L2Norm(nn.Module):
    def forward(self, x):
        return x.norm()


class SingleWavelet(nn.Module):
    def __init__(self, xi, T):
        """xi: center of the Morlet wavelet
        T: size of temporal support.
        """

        super(SingleWavelet, self).__init__()
        sigma = fb.compute_sigma_psi(xi, 1)
        psi_hat = fb.morlet1D(T, xi, sigma)  # define wavelet centered at xi
        psi_hat = nn.Parameter(cplx.from_numpy(psi_hat), requires_grad=False)
        self.register_parameter('psi_hat', psi_hat)

    def forward(self, x):
        x_hat = fft1d_c2c(x)
        x_filt_hat = cplx.mul(x_hat, self.psi_hat)
        # x_filt = ifft1d_c2c_normed(x_filt_hat)

        return x_filt_hat


class Wavelet(nn.Module):
    def __init__(self, N, Q, T):
        """N : number of octaves
        Q: number of wavelets per octave
        T: size of temporal support
        """
        super(Wavelet, self).__init__()
        self.N = N
        self.Q = Q
        self.T = T
        xi, sigma = self.compute_wavelet_parameters()
        self.xi = xi
        self.sigma = sigma
        psi_hat = self.compute_wavelet_filters()
        psi_hat = nn.Parameter(cplx.from_numpy(psi_hat), requires_grad=False)
        self.register_parameter('psi_hat', psi_hat)

    def compute_wavelet_parameters(self):
        # compute wavelets starting from the high frequencies
        # xi, sigma = fb.compute_params_filterbank(0, Q, )
        xi_curr = fb.compute_xi_max(self.Q)  # initialize at max possible xi
        r_psi = np.sqrt(0.5)
        sigma_curr = fb.compute_sigma_psi(xi_curr, self.Q, r=r_psi)  # corresponds to xi_curr

        xi, sigma = [], []
        factor = 1. / math.pow(2., 1. / self.Q)
        for nq in range(self.N * self.Q):
            xi.append(xi_curr)
            sigma.append(sigma_curr)
            xi_curr *= factor
            sigma_curr *= factor

        return xi, sigma

    def compute_wavelet_filters(self):
        psi_hat = [fb.morlet1D(self.T, xi, sigma)
                   for xi, sigma in zip(self.xi, self.sigma)]
        psi_hat = np.stack(psi_hat, 0)
        return psi_hat

    def forward(self, x):
        x_hat = fft1d_c2c(x).unsqueeze(2)
        x_filt_hat = cplx.mul(x_hat, self.psi_hat)

        return x_filt_hat.norm(dim=-1).norm(dim=-1) ** 2


class PhaseHarmonicTransform(nn.Module):
    def __init__(self, N, Q, K, T, k_type='linear',
                 wav_type="battle_lemarie", high_freq=0.5,
                 check_for_nan=False, anti_aliasing=False):
        """
        First order of the Phase Harmonics Transform. The forward method returns a
        channel-collapsed representation, whereas the compute_phase_harmonics method
        returns the full vector with more details about the coefficients' origin.

        Parameters:
            N : number of octaves
            Q: number of wavelets per octave
            K: number of phase harmonics
            T: size of temporal support
            wav_type: type of wavelets, 'battle_lemarie' or 'morlet'
            check_for_nan: debug option to track NaNs, set to False by default

        Input:
            x: input signal to be transformed, with shape (batch, channels, time, Re+Im)

        forward Output:
            x_transf: phase harmonics transform of x with
                shape (batch, K * #wavelets * channel, time, Re + Im)

        comput_phase_harmonics Output:
            x_exp: phase harmonics transform of x with
                shape (batch, K, #wavelets, channel, time, Re + Im)
        """

        super(PhaseHarmonicTransform, self).__init__()
        self.N = N
        self.Q = Q
        self.T = T
        self.K = K
        self.k_type = k_type
        self.wav_type = wav_type
        self.phase_exp = PhaseExp(self.K, k_type=k_type,
                                  keep_k_dim=True, check_for_nan=check_for_nan)
        self.check_for_nan = check_for_nan
        self.anti_aliasing = anti_aliasing

        max_freq_comput = high_freq * 3 * (self.K - 1)
        dilation_pow2 = int(max(np.ceil(np.log2(max_freq_comput)), 0))
        self.upsample = 2 ** dilation_pow2

        # initialize wavelets
        if self.wav_type == 'morlet':
            xi, sigma, sigma_low = lfb.compute_morlet_parameters(self.N, self.Q)
        elif self.wav_type == 'battle_lemarie':
            if Q != 1:
                print("\nWarning: width of Battle-Lemarie wavelets not adaptative with Q in the current implementation.\n")
            xi, sigma = lfb.compute_battle_lemarie_parameters(self.N, self.Q, high_freq=high_freq)
        elif self.wav_type == 'bump_steerable':
            if Q != 1:
                print("\nWarning: width of Bump-Steerable wavelets not adaptative with Q in the current implementation.\n")
            xi, sigma = lfb.compute_bump_steerable_parameters(self.N, self.Q, high_freq=high_freq)
        else:
            raise ValueError("Unkown wavelet type: {}".format(self.wav_type))
        self.xi = xi
        self.sigma = sigma
        psi_hat = self.compute_wavelet_filters(self.xi, self.sigma)

        # initialize low-pass
        if self.wav_type == 'morlet':
            self.sigma.append(sigma_low)
        self.xi.append(0)
        phi_hat = self.compute_low_pass()

        # join wavelets and low-pass into a filter bank (low-frequencies at the end)
        filt_hat = np.concatenate((psi_hat, phi_hat), axis=0)
        # upsample
        if self.anti_aliasing:
            hs = self.T // 2

            filt_pos, filt_neg = filt_hat[:, :hs], filt_hat[:, hs:]
            zeros = [np.zeros_like(filt_pos)] * (self.upsample - 1)
            filt_hat = np.concatenate([filt_pos] + zeros + [filt_neg] + zeros, axis=1)

            aaf = lfb.compute_anti_aliasing_filt(self.T, 5)
            aaf = nn.Parameter(cplx.from_numpy(aaf[None, None, None, :]), requires_grad=False)
            self.register_parameter('aaf', aaf)

        # pytorch parameter
        filt_hat = nn.Parameter(cplx.from_numpy(filt_hat), requires_grad=False)
        self.register_parameter('filt_hat', filt_hat)

    def compute_wavelet_filters(self, xis, sigmas):
        """Computes the wavelets Fourier transforms given their parameters
        in xis and sigmas.
        """

        if self.wav_type == "morlet":
            psi_hat = [fb.morlet1D(self.T, xi, sigma) for xi, sigma in zip(xis, sigmas)]
        elif self.wav_type == "battle_lemarie":
            # psi_hat = [lfb.battle_lemarie_psi(self.T, self.Q, xi) for xi in xis]
            psi_hat = [lfb.battle_lemarie_psi(self.T, 1, xi) / np.sqrt(self.Q) for xi in xis]
        elif self.wav_type == "bump_steerable":
            psi_hat = [lfb.bump_steerable_psi(self.T, 1, xi) / np.sqrt(self.Q) for xi in xis]
        psi_hat = np.stack(psi_hat, 0)
        return psi_hat

    def compute_low_pass(self):
        """Compute the low-pass Fourier transforms assuming it has the same variance
        as the lowest-frequency wavelet.
        """
        if self.wav_type == "morlet":
            sigma_low = self.sigma[-1]
            phi_hat = fb.gauss1D(self.T, sigma_low)
        elif self.wav_type == "battle_lemarie":
            xi_low = self.xi[-2]
            # phi_hat = lfb.battle_lemarie_phi(self.T, self.Q, xi_low)
            phi_hat = lfb.battle_lemarie_phi(self.T, 1, xi_low)
        elif self.wav_type == "bump_steerable":
            xi_low = self.xi[-2]
            # phi_hat = lfb.battle_lemarie_phi(self.T, self.Q, xi_low)
            phi_hat = lfb.bump_steerable_phi(self.T, 1, xi_low)
        return phi_hat[None, :]

    def compute_wavelet_coefficients(self, x):
        # filter x with wavelets and low-pass in self.filt_hat
        x_hat = fft1d_c2c(x)
        # upsample
        if self.anti_aliasing:
            len2add = self.upsample - 1
            zeros = torch.zeros_like(x_hat)[:, :, :self.T // 2, :]
            x_pos, x_neg = x_hat[:, :, :self.T // 2], x_hat[:, :, self.T // 2:]
            x_hat = torch.cat(
                [x_pos] + [zeros] * len2add + [x_neg] + [zeros] * len2add, dim=2)

        # apply wavelet filters
        print('x_hat shape',x_hat.shape)
        print('filt_hat shape',self.filt_hat.shape)
        x_filt_hat = cplx.mul(x_hat.unsqueeze(1), self.filt_hat.unsqueeze(0).unsqueeze(2))

        if self.anti_aliasing:
            hs, ups = self.T // 2, self.upsample
            x_filt_pos = x_filt_hat[..., :hs, :]
            x_filt_neg = x_filt_hat[..., hs * ups:hs * (ups + 1), :]
            x_filt_hat = torch.cat((x_filt_pos, x_filt_neg), dim=-2)
            x_filt_hat = cplx.mul(x_filt_hat, self.aaf)

        x_filt = ifft1d_c2c_normed(x_filt_hat)
        return x_filt

    def compute_phase_harmonics(self, x):

        x_filt = self.compute_wavelet_coefficients (x)

        # phase harmonics
        x_exp = self.phase_exp(x_filt)

        # for debug, can be ignored
        if x.requires_grad and self.check_for_nan:
            x_hat.register_hook(HookDetectNan('x_hat in PhaseHarmonicTransform.compute_phase_harmonics'))
            x_filt_hat.register_hook(HookDetectNan('x_filt_hat in PhaseHarmonicTransform.compute_phase_harmonics'))
            x_filt.register_hook(HookDetectNan('x_filt in PhaseHarmonicTransform.compute_phase_harmonics'))
            x_exp.register_hook(HookDetectNan('x_exp in PhaseHarmonicTransform.compute_phase_harmonics'))

        return x_exp

    def forward(self, x):
        x_exp = self.compute_phase_harmonics(x)

        # collapse dimensions k, lambda, channel together
        s = x.size()
        x_transf = x_exp.view(s[0], -1, *s[2:])

        return x_transf


class PhaseHarmonicAvg(PhaseHarmonicTransform):
    """First Order moments of Phase Harmonics Transform."""

    def compute_first_order(self, x):
        x_exp = self.compute_phase_harmonics(x)
        x_fst = x_exp.mean(dim=-2)
        return x_fst

    def forward(self, x):
        fst_order = self.compute_first_order(x)

        batch = fst_order.size(0)
        x_fst = fst_order.view(batch, -1, 2)

        return x_fst

class PhaseHarmonicCov(PhaseHarmonicTransform):
    """
    Phase Harmonics Covariance embedding. Embedding is the covariance of the output
    of PhaseHarmonicTransform.compute_phase_exponent.
    A number of coefficients are discarded, especially the coefficients such that
        |k * xi_lambda - k' * xi_{lambda'}| + k * Delta_lambda + k' * Delta_{lambda'} >= 1
    which suffer aliasing problems. Delta_lambda is defined from sigma_lambda
    with multiplicative factor sigma2delta.

    Parameters:
        *args and **kwargs : parameters for PhaseHarmonicTransform
        sigma2delta: Delta_lambda = sigma2delta * sigma_lambda
        ndiag : number of diagonals to keep. Only Cov[k, k', lambda, lambda']
            where |lambda - lambda'| < ndiag are kept.
        fst_order : returns the first order as well as the covariance
        tolerance : multiplicative factor to discard coefficients without energy.
            Embedding only contains Cov[k, k', lambda, lambda'] s.t.
            |k * xi_lambda - k' * xi_{lambda'}| <= tolerance * (k * Delta_lambda + k' * Delta_{lambda'})
        multi_same : keeps coefficients Cov(k, k, lambda, lambda) where k > 0
        max_chunk : cuts embedding into chunks of maximal size max_chunk during
            computation in order to reduce GPU memory.

    Inputs:
        x : input signal to be transformed, shape (batch, channels, time, Re+Im)
        fst_order : first order moment from signal to be reconstructed x_0

    forward Outputs:
        x_moments : (possibly first and) second moments of phase harmonic transform.
            shape (batch, channels', Re+Im) for first order moments and
            (batch, channels'', Re+Im) for second order moments.
    """

    def __init__(self, *args, sigma2delta=(0.5 * np.pi / np.sqrt(3)), ndiag=2,
                 fst_order=True, tolerance=2., multi_same=True, max_chunk=2000,
                 zero_fst=True, **kwargs):
        super(PhaseHarmonicCov, self).__init__(*args, **kwargs)

        self.fst_order = fst_order
        if ndiag is None:
            ndiag = len(self.xi)
        self.ndiag = ndiag
        self.tol = tolerance
        self.multi_same = multi_same
        self.max_chunk = max_chunk
        self.zero_fst = zero_fst

        xi0 = self.xi[0]  # center of highest frequency wavelet
        if self.wav_type == "morlet":
            sigma0 = self.sigma[0]  # variance of highest frequency wavelet
            delta0 = sigma2delta * sigma0  # wavelet's energy is within [xi0 - delta0, xi0 + delta0]
        elif self.wav_type == "battle_lemarie":
            delta0 = xi0 / 3
        elif self.wav_type == "bump_steerable":
            delta0 = xi0 * lfb.BS_xi2sigma
        self.xi2delta = delta0 / xi0  # wavelet's energy is within [xi0 - delta0, xi0 + delta0]

        xi_idx, k_idx = self.compute_idx_info()
        self.xi_idx = torch.LongTensor(xi_idx)
        self.k_idx = torch.LongTensor(k_idx)

        idx = len(self.xi) * k_idx + xi_idx
        nb_idx = idx.shape[0]
        nb_chunk = int(math.ceil(nb_idx / self.max_chunk))
        idxs = np.array_split(idx, nb_chunk, axis=0)
        self.idxs = [Variable(torch.LongTensor(idx)) for idx in idxs]

    def cuda(self):
        super(PhaseHarmonicCov, self).cuda()
        self.idxs = [idx.cuda() for idx in self.idxs]
        self.xi_idx = self.xi_idx.cuda()
        self.k_idx = self.k_idx.cuda()
        return self

    def cpu(self):
        super(PhaseHarmonicCov, self).cpu()
        self.idxs = [idx.cpu() for idx in self.idxs]
        self.xi_idx = self.xi_idx.cpu()
        self.k_idx = self.k_idx.cpu()
        return self

    def shape(self):
        """Returns the size factor between input and output."""
        o2_s = sum(idx.size(0) for idx in self.idxs)
        if self.fst_order:
            o1_s = len(self.xi) * (1 if self.zero_fst else self.K)
            return o1_s + o2_s
        else:
            return o2_s

    def shape_effect(self):
        """Returns the number of non-aliased coefficients computed."""
        return self.shape()

    def compute_idx_info(self):
        n_lambda = len(self.xi)
        xi0 = self.xi[0]  # center of highest frequency wavelet
        xi2delta = self.xi2delta  # wavelet's energy is within [xi0 - delta0, xi0 + delta0]

        if self.k_type == 'linear':
            k = np.arange(self.K)
        elif self.k_type == 'log2':
            k = np.concatenate(([0], np.power(2, np.arange(self.K - 1))))
        xi = np.array(self.xi)
        kxi = k[:, None] * xi[None, :]  # size (K, N * Q + 1)

        # size in fourier is about max(1, k) * delta where delta is the size of x * psi
        delta = np.maximum(k, 1)[:, None] * xi[None, :] * xi2delta
        # low-pass has the same variance as the lowest frequency wavelet
        delta[:, -1] = delta[:, -2]

        center_freq_dist = np.abs(kxi[:, None, :, None] - kxi[None, :, None, :])
        energy_width = delta[:, None, :, None] + delta[None, :, None, :]

        # non aliasing coefficients
        non_aliasing = center_freq_dist + energy_width <= 1.

        # coefficients with energy
        has_energy = center_freq_dist <= energy_width * self.tol

        # j' >= j
        freq_sup = xi[None, None, :, None] - xi[None, None, None, :] > 0
        freq_eq = xi[None, None, :, None] - xi[None, None, None, :] == 0
        freq_eq = np.logical_and(freq_eq, k[:, None, None, None] - k[None, :, None, None] <= 0)
        triang_sup = np.logical_or(freq_eq, freq_sup)

        # kept diagonal coefficients
        j = np.arange(n_lambda)
        keep_diag = np.abs(j[None, None, :, None] - j[None, None, None, :]) < self.ndiag

        # get indices for xi and k
        do_keep_idx = np.logical_and(non_aliasing, has_energy)
        do_keep_idx = np.logical_and(do_keep_idx, triang_sup)
        do_keep_idx = np.logical_and(do_keep_idx, keep_diag)
        # remove [k, k, lambda, lambda] if asked
        if not self.multi_same:
            sk, slam = do_keep_idx.shape[0], do_keep_idx.shape[2]
            for k, lam in product(range(1, sk),range(slam)):
                do_keep_idx[k, k, lam, lam] = False
        # get used indices
        keep_idx = np.argwhere(do_keep_idx)

        keep_idx_k = keep_idx[:, :2]
        keep_idx_xi = keep_idx[:, 2:]

        return keep_idx_xi, keep_idx_k

    def compute_first_order(self, x):
        x_exp = self.compute_phase_harmonics(x)
        fst_order = x_exp.mean(dim=-2).unsqueeze(-2)
        if self.zero_fst:
            fst_order[:, 1:] = 0
        return fst_order

    def compute_phase_harmonics_moments(self, x, fst_order):
        x_exp = self.compute_phase_harmonics(x)
        # size (batch, K, #wavelets, channel, time, Re + Im)

        x_exp_off = x_exp - fst_order
        if self.fst_order:
            fst_order = x_exp.mean(dim=-2).unsqueeze(-2)
            if self.zero_fst:
                fst_order[:, 1:] = 0
            # size (batch, K, N * Q + 1, channel, 1, Re + Im)

        s = x_exp_off.size()
        x_exp_off = x_exp_off.view(s[0], s[1] * s[2], *s[3:])

        scd_order = []
        for idx in self.idxs:
            idx0 = idx[:, 0]
            idx1 = idx[:, 1]

            x_exp_0 = torch.index_select(x_exp_off, 1, idx0)
            x_exp_1 = torch.index_select(x_exp_off, 1, idx1)
            # size (batch, elt_kept, channel, time, Re + Im)

            x_cov = cplx.mul(x_exp_0, cplx.conjugate(x_exp_1))
            # size (batch, elt_kept, channel, time, Re + Im)
            scd_o = torch.mean(x_cov, dim=-2)
            # size (batch, elt_kept, channel, Re + Im)
            scd_order.append(scd_o)
        scd_order = torch.cat(scd_order, dim=1)

        if x.requires_grad and self.check_for_nan:
            x_exp.register_hook(HookDetectNan('x_exp in  ' + self.__class__.__name__ + '.compute_phase_harmonics_moments'))
            x_exp_off.register_hook(HookDetectNan('x_exp_off in  ' + self.__class__.__name__ + '.compute_phase_harmonics_moments'))
            x_exp_1.register_hook(HookDetectNan('x_exp_1 in  ' + self.__class__.__name__ + '.compute_phase_harmonics_moments'))
            x_exp_0.register_hook(HookDetectNan('x_exp_0 in  ' + self.__class__.__name__ + '.compute_phase_harmonics_moments'))
            x_cov.register_hook(HookDetectNan('x_cov in  ' + self.__class__.__name__ + '.compute_phase_harmonics_moments'))
            if self.fst_order:
                fst_order.register_hook(HookDetectNan('fst_order in  ' + self.__class__.__name__ + '.compute_phase_harmonics_moments'))
            scd_o.register_hook(HookDetectNan('x_cov_mean in  ' + self.__class__.__name__ + '.compute_phase_harmonics_moments'))

        if self.fst_order:
            if self.zero_fst:
                fst_order = fst_order[:, :1]
            return (fst_order, scd_order)
        else:
            return (scd_order,)

    def forward(self, x, fst_order):
        batch = x.size(0)
        emb = self.compute_phase_harmonics_moments(x, fst_order)
        if self.fst_order:
            fst_order, scd_order = emb
            fst_order = fst_order.squeeze(3).squeeze(-2).squeeze(1)
            scd_order = scd_order.view(batch, -1, 2)
            x_moments = (fst_order, scd_order)
        else:
            scd_order, = emb
            scd_order = scd_order.view(batch, -1, 2)
            x_moments = (scd_order,)

        if x.requires_grad and self.check_for_nan:
            if self.fst_order:
                fst_order.register_hook(HookDetectNan('fst_order in  ' + self.__class__.__name__ + '.forward'))
            scd_order.register_hook(HookDetectNan('scd_order in  ' + self.__class__.__name__ + '.forward'))
            x_moments.register_hook(HookDetectNan('x_moments in  ' + self.__class__.__name__ + '.forward'))

        return x_moments


class PhaseHarmonicCovNonLinCoeff(PhaseHarmonicCov):
    def __init__(self, *args, M=None, **kwargs):
        super(PhaseHarmonicCovNonLinCoeff, self).__init__(*args, **kwargs)
        if isinstance(M, float):
            M = int(sum(idx.size(0) for idx in self.idxs) * M)
        self.M = M
        self.energy_ratio = 1. if M is None else None

    def shape(self):
        """Returns the size factor between input and output."""
        o2_s = sum(idx.size(0) for idx in self.idxs)
        if self.M is not None:
            o2_s = min(o2_s, self.M)
        if self.fst_order:
            o1_s = len(self.xi) * (1 if self.zero_fst else self.K)
            return o1_s + o2_s
        else:
            return o2_s

    def non_linear_coefficient_selection(self, x0, fst_order):
        if self.M is None:  # keep all coefficients, same as PhaseHarmonicCov
            return

        # compute embedding of true signal
        emb0 = self(x0, fst_order)
        if self.fst_order:
            _, scd_order = emb0
        else:
            scd_order, = emb0

        # compute energy contribution of each coefficient
        coeff_energy = (cplx.modulus(scd_order)[0] ** 2).data.cpu().numpy()
        max_energy_idxs = np.argsort(coeff_energy)[-self.M:]
        ratio = np.sum(coeff_energy[max_energy_idxs]) / np.sum(coeff_energy)
        self.energy_ratio = ratio

        # extract self.M coefficients with most energy
        self.xi_idx = self.xi_idx[max_energy_idxs]
        self.k_idx = self.k_idx[max_energy_idxs]
        idx = len(self.xi) * self.k_idx + self.xi_idx
        nb_idx = idx.shape[0]
        nb_chunk = int(math.ceil(nb_idx / self.max_chunk))
        idxs = np.array_split(idx, nb_chunk, axis=0)
        is_cuda = self.idxs[0].is_cuda
        self.idxs = [Variable(torch.LongTensor(idx)) for idx in idxs]
        if is_cuda:
            self.idxs = [idx.cuda() for idx in self.idxs]


class PhaseHarmonicPrunedBase(nn.Module):
    '''
    Abstract base class for pruned phase harmonics
    '''
    def __init__(self, N, Q, T, wav_type="battle_lemarie",
                 high_freq=0.5, delta_j=1, delta_k=[-1, 0, 1],
                 num_k_modulus=3, delta_cooc=2, zero_fst=True,
                 max_chunk=None, check_for_nan=False):
        super(PhaseHarmonicPrunedBase, self).__init__()

        self.N = N
        self.Q = Q
        self.T = T
        self.wav_type = wav_type
        self.high_freq = high_freq
        self.phase_harmonics = PhaseHarmonic(check_for_nan=check_for_nan)
        self.check_for_nan = check_for_nan
        self.init_filters()

        self.delta_j = delta_j if delta_j is not None else len(self.xi)
        self.delta_k = delta_k
        self.num_k_modulus = num_k_modulus
        self.delta_cooc = delta_cooc
        self.zero_fst = zero_fst

        self.max_chunk = max_chunk

    def init_filters(self):
        Q = self.Q
        high_freq = self.high_freq
        # initialize wavelets
        if self.wav_type == 'morlet':
            xi, sigma, sigma_low = lfb.compute_morlet_parameters(self.N, self.Q, analytic=True)
        elif self.wav_type == 'battle_lemarie':
            if Q != 1:
                print("\nWarning: width of Battle-Lemarie wavelets not adaptative with Q in the current implementation.\n")
            xi, sigma = lfb.compute_battle_lemarie_parameters(self.N, self.Q, high_freq=high_freq)
        elif self.wav_type == 'bump_steerable':
            if Q != 1:
                print("\nWarning: width of Bump-Steerable wavelets not adaptative with Q in the current implementation.\n")
            xi, sigma = lfb.compute_bump_steerable_parameters(self.N, self.Q, high_freq=high_freq)
        else:
            raise ValueError("Unkown wavelet type: {}".format(self.wav_type))
        self.xi = xi
        self.sigma = sigma
        psi_hat = self.compute_wavelet_filters(self.xi, self.sigma)

        # initialize low-pass
        if self.wav_type == 'morlet':
            self.sigma.append(sigma_low)
        self.xi.append(0)
        phi_hat = self.compute_low_pass()

        # join wavelets and low-pass into a filter bank (low-frequencies at the end)
        filt_hat = np.concatenate((psi_hat, phi_hat), axis=0)

        # pytorch parameter
        filt_hat = nn.Parameter(cplx.from_numpy(filt_hat), requires_grad=False)
        self.register_parameter('filt_hat', filt_hat)

    def compute_wavelet_filters(self, xis, sigmas):
        """Computes the wavelets Fourier transforms given their parameters
        in xis and sigmas.
        """

        if self.wav_type == "morlet":
            psi_hat = [fb.morlet1D(self.T, xi, sigma) for xi, sigma in zip(xis, sigmas)]
        elif self.wav_type == "battle_lemarie":
            # psi_hat = [lfb.battle_lemarie_psi(self.T, self.Q, xi) for xi in xis]
            psi_hat = [lfb.battle_lemarie_psi(self.T, 1, xi) / np.sqrt(self.Q) for xi in xis]
        elif self.wav_type == "bump_steerable":
            psi_hat = [lfb.bump_steerable_psi(self.T, 1, xi) / np.sqrt(self.Q) for xi in xis]
        psi_hat = np.stack(psi_hat, 0)

        return psi_hat

    def compute_low_pass(self):
        """Compute the low-pass Fourier transforms assuming it has the same variance
        as the lowest-frequency wavelet.
        """
        if self.wav_type == "morlet":
            sigma_low = self.sigma[-1]
            phi_hat = fb.gauss1D(self.T, sigma_low)
        elif self.wav_type == "battle_lemarie":
            xi_low = self.xi[-2]
            # phi_hat = lfb.battle_lemarie_phi(self.T, self.Q, xi_low)
            phi_hat = lfb.battle_lemarie_phi(self.T, 1, xi_low)
        elif self.wav_type == "bump_steerable":
            xi_low = self.xi[-2]
            # phi_hat = lfb.battle_lemarie_phi(self.T, self.Q, xi_low)
            phi_hat = lfb.bump_steerable_phi(self.T, 1, xi_low)
        return phi_hat[None, :]

    def compute_wavelet_coefficients(self, x):
        # filter x with wavelets and low-pass in self.filt_hat
        x_hat = fft1d_c2c(x)

        # apply wavelet filters
        print('x_hat shape',x_hat.shape)
        print('filt_hat shape',self.filt_hat.shape)
        x_filt_hat = cplx.mul(x_hat.unsqueeze(2), self.filt_hat.unsqueeze(0).unsqueeze(1))

        x_filt = ifft1d_c2c_normed(x_filt_hat)
        return x_filt

    def shape(self):
        """Returns the number of complex coefficients in the embedding."""
        raise NotImplementedError

    def num_coeff(self):
        """Returns the effective number of (real) coefficients in the embedding"""
        raise NotImplementedError

    def compute_idx_info(self, num_k_modulus, delta_k):
        raise NotImplementedError

    def balance_chunks(self, *args):
        """Cuts all torch tensors in args in corresponding balanced chunks, along
        their first dimension.
        For each input tensor, the output is a list of chunks of this tensor.
        """

        if self.max_chunk is None:
            return [[a] for a in args]  # each tensor is divided in only one chunk

        n_idx = args[0].shape[0]

        n_stops = int(np.ceil(n_idx / self.max_chunk))
        base_size, leftover = n_idx // n_stops, n_idx % n_stops
        sizes = [base_size + int(i < leftover) for i in range(n_stops)]
        stops_pos = np.cumsum([0] + sizes)

        chunked_args = []
        for tens in args:
            chunked_tens = [tens[s:e] for s, e in zip(stops_pos[:-1], stops_pos[1:])]
            chunked_args.append(chunked_tens)

        return chunked_args

class PhaseHarmonicPruned(PhaseHarmonicPrunedBase):
    def __init__(self, *args, **kwargs):
        super(PhaseHarmonicPruned, self).__init__(*args, **kwargs)

        xi_idx, ks, num_coeff_abs, num_coeff_cplx = self.compute_idx_info(
            self.num_k_modulus, self.delta_k)
        self.num_coeff_abs = num_coeff_abs
        self.num_coeff_cplx = num_coeff_cplx

        nb_idx = xi_idx.shape[0]
        nb_chunk = 1 if self.max_chunk is None else int(math.ceil(nb_idx / self.max_chunk))
        xi_idx = np.array_split(xi_idx, nb_chunk, axis=0)
        ks = np.array_split(ks, nb_chunk, axis=0)
        self.xi_idx = [torch.LongTensor(xi_id) for xi_id in xi_idx]
        self.ks = [torch.LongTensor(k) for k in ks]

    def cuda(self):
        super(PhaseHarmonicPrunedBase, self).cuda()
        self.xi_idx = [xi_idx.cuda() for xi_idx in self.xi_idx]
        self.ks = [ks.cuda() for ks in self.ks]
        return self

    def cpu(self):
        super(PhaseHarmonicPrunedBase, self).cpu()
        self.xi_idx = [xi_idx.cpu() for xi_idx in self.xi_idx]
        self.ks = [ks.cpu() for ks in self.ks]
        return self

    def shape(self):
        """Returns the number of complex coefficients in the embedding."""
        o2_s = sum(idx.size(0) for idx in self.xi_idx)
        o1_s = len(self.xi) * (1 if self.zero_fst else self.K)
        return o1_s, o2_s

    def num_coeff(self):
        """Returns the effective number of (real) coefficients in the embedding"""
        J = len(self.xi)

        # first order coefficients (complex)
        o1 = J

        # second order coefficients (some are complex)
        o2 = self.num_coeff_abs + 2 * self.num_coeff_cplx

        return o1 + o2

    def compute_idx_info(self, num_k_modulus, delta_k):
        if delta_k is None:
            num_k_modulus = min(num_k_modulus, 2)

        J = len(self.xi)
        xi = np.array(self.xi)
        num_coeff_abs = 0  # number of coefficients which are already real
        num_coeff_cplx = 0  # number of truly complex coefficients

        # compute coeffs j = j'
        jeq = np.arange(np.size(xi))
        jeq0, jpeq0, jeq1, jpeq1 = jeq, jeq, jeq, jeq
        keq0, kpeq0 = np.zeros_like(jeq0), np.zeros_like(jpeq0)
        keq1, kpeq1 = np.zeros_like(jeq0), np.ones_like(jpeq0)

        num_coeff_abs += jeq.size
        num_coeff_cplx += jeq.size

        # compute coefficients j < j'
        j, jp = np.where(xi[:, None] > xi[None, :])  # j < j'
        loc = np.where(jp - j <= self.delta_j * self.Q)  # |j - j'| < delta_j (* Q)
        j, jp = j[loc], jp[loc]

        # compute <|x*psi_j|, [x*psi_j']^k>
        kp0 = np.ravel(np.repeat(np.arange(num_k_modulus)[None, :], j.size, axis=0))
        k0 = np.zeros_like(kp0)
        j0 = np.ravel(np.repeat(j[:, None], num_k_modulus, axis=1))
        jp0 = np.ravel(np.repeat(jp[:, None], num_k_modulus, axis=1))

        num_coeff_abs += j.size
        num_coeff_cplx += j.size * (num_k_modulus - 1)

        # compute <x*psi_j, [x*psi_j']^(2^(j-j') +- 1)>
        # num_k = 2 * self.K + 1
        if delta_k is None:  # only compute k=1, k'=1
            num_k = 1
            k1 = np.ones_like(j[:])
            kp1 = np.ones_like(j[:])
            j1 = j.copy()
            jp1 = jp.copy()

        else:
            num_k = len(delta_k)
            delta_k = np.array(delta_k)

            # center = (jp - j)[:, None] / self.Q + .2 * delta_k[None, :]
            center = (jp - j)[:, None] / self.Q
            kp1 = np.ravel(np.power(2., center) + delta_k[None, :])  # moves j' to j
            k1 = np.ones_like(kp1)
            j1 = np.ravel(np.repeat(j[:, None], num_k, axis=1))
            jp1 = np.ravel(np.repeat(jp[:, None], num_k, axis=1))


        num_coeff_cplx += j.size * num_k

        j = np.concatenate((jeq0, jeq1, j0, j1))
        jp = np.concatenate((jpeq0, jpeq1, jp0, jp1))
        k = np.concatenate((keq0, keq1, k0, k1))
        kp = np.concatenate((kpeq0, kpeq1, kp0, kp1))

        keep_k = np.stack((k, kp), axis=1)
        keep_idx_xi = np.stack((j, jp), axis=1)


        # k fractionnary would lead to discontinuity
        keep_k = np.floor(keep_k).astype(int)

        return keep_idx_xi, keep_k, num_coeff_abs, num_coeff_cplx


    def forward(self, x):
        # # filter x with wavelets and low-pass in self.filt_hat
        # x_hat = fft1d_c2c(x)

        # # apply wavelet filters
        # x_filt_hat = cplx.mul(x_hat.unsqueeze(2), self.filt_hat.unsqueeze(0).unsqueeze(1))

        # x_filt = ifft1d_c2c_normed(x_filt_hat)
        x_filt = self.compute_wavelet_coefficients(x)

        # first order
        k0 = torch.zeros_like(x_filt[0, 0, ..., 0, 0]).view(-1)[:x_filt.size(2)].long()
        fst_order = self.phase_harmonics(x_filt, k0)
        for spatial_dim in x.size()[2:-1]:
            fst_order = torch.mean(fst_order, dim=-2)

        # second order
        scd_order = []
        for xi_idx, ks in zip(self.xi_idx, self.ks):
            x_filt0 = torch.index_select(x_filt, 2, xi_idx[:, 0])
            x_filt1 = torch.index_select(x_filt, 2, xi_idx[:, 1])
            k0, k1 = ks[:, 0], ks[:, 1]

            scd_0 = self.phase_harmonics(x_filt0, k0)
            scd_1 = self.phase_harmonics(x_filt1, -k1)
            scd = torch.mean(cplx.mul(scd_0, scd_1), dim=-2)
            scd_order.append(scd)
        scd_order = torch.cat(scd_order, dim=2)

        # for debug, can be ignored
        if x.requires_grad and self.check_for_nan:
            x_filt.register_hook(HookDetectNan('x_filt in PhaseHarmonicTransform.compute_phase_harmonics'))
            fst_order.register_hook(HookDetectNan('fst_order in PhaseHarmonicTransform.compute_phase_harmonics'))
            scd_order.register_hook(HookDetectNan('scd_order in PhaseHarmonicTransform.compute_phase_harmonics'))

        return fst_order, scd_order


class PhaseHarmonicPrunedSelect(PhaseHarmonicPrunedBase):
    def __init__(self, *args, coeff_select=['harmonic', 'mixed'], **kwargs):
        super(PhaseHarmonicPrunedSelect, self).__init__(*args, **kwargs)

        self.coeff_select = coeff_select

        self.idx_info = self.compute_idx_info(self.num_k_modulus, self.delta_k)

        self.num_coeff_abs = 0
        self.num_coeff_cplx = 0
        for ctype in coeff_select:
            self.num_coeff_abs += self.idx_info[ctype]['ncoef_real']
            self.num_coeff_cplx += self.idx_info[ctype]['ncoef_cplx']

        self.xi_idx = [torch.LongTensor(self.idx_info[ctype]['xi_idx'])
                       for ctype in coeff_select]
        self.ks = [torch.LongTensor(self.idx_info[ctype]['k'])
                   for ctype in coeff_select]

        self.is_cuda = False

        # TODO: implement chunks

        # xi_idx, ks, num_coeff_abs, num_coeff_cplx = self.compute_idx_info()
        # self.num_coeff_abs = num_coeff_abs
        # self.num_coeff_cplx = num_coeff_cplx

        # nb_idx = xi_idx.shape[0]
        # nb_chunk = 1 if self.max_chunk is None else int(math.ceil(nb_idx / self.max_chunk))
        # xi_idx = np.array_split(xi_idx, nb_chunk, axis=0)
        # ks = np.array_split(ks, nb_chunk, axis=0)
        # self.xi_idx = [torch.LongTensor(xi_id) for xi_id in xi_idx]
        # self.ks = [torch.LongTensor(k) for k in ks]

    def cuda(self):
        super(PhaseHarmonicPrunedSelect, self).cuda()
        self.xi_idx = [x.cuda() for x in self.xi_idx]
        self.ks = [k.cuda() for k in self.ks]
        self.is_cuda = True
        return self

    def cpu(self):
        super(PhaseHarmonicPrunedSelect, self).cpu()
        self.xi_idx = [x.cpu() for x in self.xi_idx]
        self.ks = [k.cpu() for k in self.ks]
        self.is_cuda = False
        return self

    def compute_idx_info(self, num_k_modulus, delta_k):
        J = len(self.xi)
        xi = np.array(self.xi)
        num_coeff_abs = 0  # number of coefficients which are already real
        num_coeff_cplx = 0  # number of truly complex coefficients

        # compute coeffs j = j'
        jeq = np.arange(np.size(xi))
        jeq0, jpeq0, jeq1, jpeq1 = jeq, jeq, jeq, jeq
        keq0, kpeq0 = np.zeros_like(jeq0), np.zeros_like(jpeq0)
        keq1, kpeq1 = np.zeros_like(jeq0), np.ones_like(jpeq0)

        num_coeff_abs += jeq.size
        num_coeff_cplx += jeq.size

        # compute coefficients j < j'
        j, jp = np.where(xi[:, None] > xi[None, :])  # j < j'
        loc = np.where(jp - j <= self.delta_j * self.Q)  # |j - j'| < delta_j (* Q)
        j, jp = j[loc], jp[loc]

        # compute <|x*psi_j|, |x*psi_j'|>  harmonic and informative modulus
        kp0 = np.zeros((j.size))
        k0 = np.zeros_like(kp0)
        j0 = np.ravel(np.repeat(j[:, None], 1, axis=1))
        jp0 = np.ravel(np.repeat(jp[:, None], 1, axis=1))

        num_coeff_abs += j.size

        # compute <x*psi_j, [x*psi_j']^(2^(j-j') +- 1)>
        # num_k = 2 * self.K + 1
        # delta_k = np.linspace(-1, 1, num=num_k)
        # print(delta_k)
        num_k = len(delta_k)
        delta_k = np.array(delta_k)

        # center = (jp - j)[:, None] / self.Q + .2 * delta_k[None, :]
        center = (jp - j)[:, None] / self.Q
        kp1 = np.ravel(np.power(2., center) + delta_k[None, :])  # moves j' to j
        k1 = np.ones_like(kp1)
        j1 = np.ravel(np.repeat(j[:, None], num_k, axis=1))
        jp1 = np.ravel(np.repeat(jp[:, None], num_k, axis=1))

        num_coeff_cplx += j.size * num_k

        j = np.concatenate((jeq0, jeq1, j0, j1))
        jp = np.concatenate((jpeq0, jpeq1, jp0, jp1))
        k = np.concatenate((keq0, keq1, k0, k1))
        kp = np.concatenate((kpeq0, kpeq1, kp0, kp1))

        keep_k = np.stack((k, kp), axis=1)
        keep_idx_xi = np.stack((j, jp), axis=1)

        # k fractionnary would lead to discontinuity
        keep_k = np.floor(keep_k).astype(int)

        dict_harm = {'k': keep_k,
                     'xi_idx': keep_idx_xi,
                     'ncoef_cplx': num_coeff_cplx,
                     'ncoef_real': num_coeff_abs}

        # Mixed coefficients:  <|x*psi_j|, [x*psi_j']^k>, k>1
        kpm = np.ravel(np.repeat(np.arange(1, num_k_modulus)[None, :], j.size, axis=0))
        km = np.zeros_like(kpm)
        jm = np.ravel(np.repeat(j[:, None], num_k_modulus-1, axis=1))
        jpm = np.ravel(np.repeat(jp[:, None], num_k_modulus-1, axis=1))

        keep_k_mix = np.stack((km, kpm), axis=1)
        keep_idx_xi_mix = np.stack((jm, jpm), axis=1)
        keep_k_mix = np.floor(keep_k_mix).astype(int)

        num_coeff_abs_mix = 0
        num_coeff_cplx_mix = j.size * (num_k_modulus - 1)

        dict_mix = {'k': keep_k_mix,
                    'xi_idx': keep_idx_xi_mix,
                    'ncoef_cplx': num_coeff_cplx_mix,
                    'ncoef_real': num_coeff_abs_mix}

        idx_info = {'harmonic':dict_harm, 'mixed':dict_mix}

        return idx_info

    def forward(self, x):
        x_filt = self.compute_wavelet_coefficients(x)

        # first order
        k0 = torch.zeros_like(x_filt[0, 0, ..., 0, 0]).view(-1)[:x_filt.size(2)].long()
        fst_order = self.phase_harmonics(x_filt, k0)
        for spatial_dim in x.size()[2:-1]:
            fst_order = torch.mean(fst_order, dim=-2)

        # second order
        scd_order = []
        # TODO: implement blocks
        for xi_idx, ks in zip(self.xi_idx, self.ks):
            x_filt0 = torch.index_select(x_filt, 2, xi_idx[:, 0])
            x_filt1 = torch.index_select(x_filt, 2, xi_idx[:, 1])
            k0, k1 = ks[:, 0], ks[:, 1]

            scd_0 = self.phase_harmonics(x_filt0, k0)
            scd_1 = self.phase_harmonics(x_filt1, -k1)
            scd = torch.mean(cplx.mul(scd_0, scd_1), dim=-2)
            scd_order.append(scd)
        if scd_order:
            scd_order = torch.cat(scd_order, dim=2)
        else:
            fst_order = Tensor([])
            scd_order = Tensor([])
            if self.is_cuda == 'cuda':
                fst_order = fst_order.cuda()
                scd_order = scd_order.cuda()

        return fst_order, scd_order

    def num_coeff(self):
        """Returns the effective number of (real) coefficients in the embedding"""

        num_coeff = 0
        for key in self.idx_info.keys():
            num_coeff += self.idx_info[key]['ncoef_real']
            num_coeff += 2 * self.idx_info[key]['ncoef_cplx']

        return num_coeff


class PhaseHarmonicPrunedSeparated(PhaseHarmonicPrunedBase):
    def __init__(self, *args, coeff_select=['harmonic', 'mixed'], **kwargs):
        super(PhaseHarmonicPrunedSeparated, self).__init__(*args, **kwargs)

        # check if valid coefficient types
        possible_coeff_select = ['harmonic', 'mixed']
        for cs in coeff_select:
            if cs not in possible_coeff_select:
                raise ValueError("Unknown coefficient type: '{}'".format(cs))
        self.coeff_select = coeff_select

        self.idx_info = self.compute_idx_info(self.num_k_modulus, self.delta_k)
        self.num_coeff_abs = 0
        self.num_coeff_cplx = 0
        for ctype in coeff_select:
            self.num_coeff_abs += self.idx_info[ctype]['ncoef_real']
            self.num_coeff_cplx += self.idx_info[ctype]['ncoef_cplx']

        if len(coeff_select) > 0:
            xi_idx = torch.cat(
                [torch.LongTensor(self.idx_info[ctype]['xi_idx']) for ctype in coeff_select],
                dim=0)
            ks = torch.cat(
                [torch.LongTensor(self.idx_info[ctype]['k']) for ctype in coeff_select],
                dim=0)
        else:
            xi_idx = torch.LongTensor([])
            ks = torch.LongTensor([])

        xi_idx_chunked, ks_chunked = self.balance_chunks(xi_idx, ks)
        self.xi_idx = xi_idx_chunked
        self.ks = ks_chunked

        self.is_cuda = False

        # TODO: implement chunks

    def cuda(self):
        super(PhaseHarmonicPrunedSeparated, self).cuda()
        self.xi_idx = [x.cuda() for x in self.xi_idx]
        self.ks = [k.cuda() for k in self.ks]
        self.is_cuda = True
        return self

    def cpu(self):
        super(PhaseHarmonicPrunedSeparated, self).cpu()
        self.xi_idx = [x.cpu() for x in self.xi_idx]
        self.ks = [k.cpu() for k in self.ks]
        self.is_cuda = False
        return self

    def compute_idx_info(self, num_k_modulus, delta_k):
        J = len(self.xi)
        xi = np.array(self.xi)
        num_coeff_abs = 0  # number of coefficients which are already real
        num_coeff_cplx = 0  # number of truly complex coefficients

        # compute coeffs j = j'
        jeq = np.arange(np.size(xi))
        jeq0, jpeq0, jeq1, jpeq1 = jeq, jeq, jeq, jeq
        keq0, kpeq0 = np.zeros_like(jeq0), np.zeros_like(jpeq0)
        keq1, kpeq1 = np.zeros_like(jeq0), np.ones_like(jpeq0)

        num_coeff_abs += jeq.size
        num_coeff_cplx += jeq.size

        # compute coefficients j < j'
        j, jp = np.where(xi[:, None] > xi[None, :])  # j < j'
        loc = np.where(jp - j <= self.delta_j * self.Q)  # |j - j'| < delta_j (* Q)
        j, jp = j[loc], jp[loc]

        # compute <x*psi_j, [x*psi_j']^k> where k is close to 2^(j-j')

        # j = 0, 1, 2, ..., j' = 1, 2, 3, ..., 2, 3, 4, ...,
        j1, jp1 = [], []
        kp1 = []
        for i1 in range(J - 1):
            ip1 = np.arange(i1 + self.Q, min(J, i1 + 1 + self.delta_j * self.Q))
            center = (ip1 - i1) / self.Q
            kp1_aux = np.power(2., center)
            kp1_round = np.round(kp1_aux)

            for ku in np.unique(kp1_round):
                idx_ku = np.argmin(np.abs(kp1_aux - ku))  # index of wavelet closest to factor ku
                j1.append(i1)
                jp1.append(ip1[idx_ku])
                kp1.append(ku)
        j1 = np.array(j1)
        jp1 = np.array(jp1)
        kp1 = np.array(kp1)
        k1 = np.ones_like(kp1)

        num_coeff_cplx += j1.size

        # compute <|x*psi_j|, |x*psi_jk|> harmonics
        k0_h = np.zeros_like(k1)
        kp0_h = np.zeros_like(kp1)
        j0_h, jp0_h = j1, jp1

        seen = set(zip(j0_h, jp0_h))  # set to remember which <|x*psi_j|, |x*psi_jk|> are accounted for
        num_coeff_abs += j0_h.size

        # compute <|x*psi_j|, |x*psi_j'|> informative modulus
        j0_im, jp0_im = np.where(xi[:, None] > xi[None, :])  # j < j'
        loc = np.where(jp0_im - j0_im <= self.delta_cooc * self.Q)  # |j - j'| < delta_cooc (* Q)
        j0_im, jp0_im = j0_im[loc], jp0_im[loc]

        keep = []
        for i in zip(j0_im, jp0_im):
            if i in seen:
                keep.append(False)
            else:
                keep.append(True)
        keep = np.array(keep)
        j0_im = j0_im[keep]
        jp0_im = jp0_im[keep]

        kp0_im = np.zeros((j0_im.size))
        k0_im = np.zeros_like(kp0_im)

        num_coeff_abs += j0_im.size


        j = np.concatenate((jeq0, jeq1, j0_h, j0_im, j1))
        jp = np.concatenate((jpeq0, jpeq1, jp0_h, jp0_im, jp1))
        k = np.concatenate((keq0, keq1, k0_h, k0_im, k1))
        kp = np.concatenate((kpeq0, kpeq1, kp0_h, kp0_im, kp1))

        keep_k = np.stack((k, kp), axis=1)
        keep_idx_xi = np.stack((j, jp), axis=1)

        # k fractionnary would lead to discontinuity
        keep_k = np.floor(keep_k).astype(int)

        dict_harm = {'k': keep_k,
                     'xi_idx': keep_idx_xi,
                     'ncoef_cplx': num_coeff_cplx,
                     'ncoef_real': num_coeff_abs}

        # Mixed coefficients:  <|x*psi_j|, [x*psi_j']^k>, k>1
        kpm = np.ravel(np.repeat(np.arange(1, num_k_modulus)[None, :], j.size, axis=0))
        km = np.zeros_like(kpm)
        jm = np.ravel(np.repeat(j[:, None], num_k_modulus-1, axis=1))
        jpm = np.ravel(np.repeat(jp[:, None], num_k_modulus-1, axis=1))

        keep_k_mix = np.stack((km, kpm), axis=1)
        keep_idx_xi_mix = np.stack((jm, jpm), axis=1)
        keep_k_mix = np.floor(keep_k_mix).astype(int)

        num_coeff_abs_mix = 0
        num_coeff_cplx_mix = j.size * (num_k_modulus - 1)

        dict_mix = {'k': keep_k_mix,
                    'xi_idx': keep_idx_xi_mix,
                    'ncoef_cplx': num_coeff_cplx_mix,
                    'ncoef_real': num_coeff_abs_mix}


        idx_info = {'harmonic':dict_harm, 'mixed':dict_mix}

        return idx_info

    def forward(self, x):
        x_filt = self.compute_wavelet_coefficients(x)

        if len(self.coeff_select) == 0:
            return Tensor([]), Tensor([])

        # first order
        k0 = torch.zeros_like(x_filt[0, 0, ..., 0, 0]).view(-1)[:x_filt.size(2)].long()
        fst_order = self.phase_harmonics(x_filt, k0)
        for spatial_dim in x.size()[2:-1]:
            fst_order = torch.mean(fst_order, dim=-2)

        # second order
        scd_order = []
        # TODO: implement blocks
        #print('x_filt shape',x_filt.shape)
        # x_filt: (1,1,filters,T,2), T is size of signal
        for xi_idx, ks in zip(self.xi_idx, self.ks):
            #print('xi_idx shape',xi_idx.shape) # (max_chunk,2)
            #print('ks shape',ks.shape) # (max_chunk,2)
            x_filt0 = torch.index_select(x_filt, 2, xi_idx[:, 0])
            #print('x_filt0 shape',x_filt0.shape) # (1,1,max_chunk,T,2)
            x_filt1 = torch.index_select(x_filt, 2, xi_idx[:, 1])
            k0, k1 = ks[:, 0], ks[:, 1]
            scd_0 = self.phase_harmonics(x_filt0, k0) # (1,1,max_chunk,T,2)
            #print('scd_0 shape',scd_0.shape)
            scd_1 = self.phase_harmonics(x_filt1, -k1)
            scd = torch.mean(cplx.mul(scd_0, scd_1), dim=-2)
            scd_order.append(scd)
        if scd_order:
            scd_order = torch.cat(scd_order, dim=2)
        else:
            fst_order = Tensor([])
            scd_order = Tensor([])
            if self.is_cuda == 'cuda':
                fst_order = fst_order.cuda()
                scd_order = scd_order.cuda()

        return fst_order, scd_order

    def num_coeff(self):
        """Returns the effective number of (real) coefficients in the embedding"""

        num_coeff = 0
        for key in self.coeff_select:
            num_coeff += self.idx_info[key]['ncoef_real']
            num_coeff += 2 * self.idx_info[key]['ncoef_cplx']

        return num_coeff
