import sys
if __name__ == "__main__":
    sys.path.append("../pyscatwave")
import math
from math import factorial as fct
import numpy as np
try:
    from scattering.scattering1d import filter_bank as fb
except ImportError:
    from scatwave.scattering1d import filter_bank as fb
from numba import jit


def compute_anti_aliasing_filt(N, p):
    freq = np.fft.fftfreq(N)
    freq[N // 2:] += 1
    aaf = np.ones_like(freq)  # initialize Anti Aliasing Filter
    idx = freq >= 0.75
    freq = freq[idx] * 4 - 3.
    freq_pow = np.power(freq, p)
    acc, a = np.zeros_like(freq), 0
    for k in range(p + 1):
        freq_pow *= freq
        bin = fct(p) / (fct(k) * fct(p - k))
        sgn = 1. if k % 2 == 0 else -1.
        acc += bin * sgn * freq_pow / (p + k + 1)
        a += bin * sgn / (p + k + 1)
    dom_coeff = -1 / a
    aaf[idx] = dom_coeff * acc + 1.
    return aaf

@jit
def compute_morlet_parameters(N, Q, analytic=False):
    sigma0 = 0.1
    sigma_low = sigma0 / math.pow(2, N)

    if analytic:
        xi_curr = fb.compute_xi_max(Q)  # initialize at max possible xi
    else:
        xi_curr = 0.5

    r_psi = np.sqrt(0.5)
    sigma_curr = fb.compute_sigma_psi(xi_curr, Q, r=r_psi)  # corresponds to xi_curr

    xi, sigma = [], []
    factor = 1. / math.pow(2., 1. / Q)

    # geometric scaling
    while sigma_curr > sigma_low:
        xi.append(xi_curr)
        sigma.append(sigma_curr)
        xi_curr *= factor
        sigma_curr *= factor

    # affine scaling
    last_xi = xi[-1]
    num_intermediate = Q - 1
    for q in range(1, num_intermediate + 1):
        factor = (num_intermediate + 1. - q) / (num_intermediate + 1.)
        xi.append(factor * last_xi)
        sigma.append(sigma_low)

    return xi, sigma, sigma_low


@jit
def compute_battle_lemarie_parameters(N, Q, high_freq=0.5):
    xi_curr = high_freq
    xi, sigma = [], []
    factor = 1. / math.pow(2., 1. / Q)
    for nq in range(N * Q):
        xi.append(xi_curr)
        xi_curr *= factor

    return xi, sigma


# BL_XI0 = 0.7593990773014584
BL_XI0 = 0.75 * 1.012470304985129


@jit
def battle_lemarie_psi(N, Q, xi):
    if Q != 1:
        raise NotImplementedError("Scaling battle-lemarie wavelets to multiple wavelets per octave not implemented yet.")
    xi0 = BL_XI0  # mother wavelet center

    # frequencies for mother wavelet with 1 wavelet per octave
    abs_freqs = np.linspace(0, 1, N + 1)[:-1]
    # frequencies for wavelet centered in xi with 1 wavelet1 per octave
    freqs = abs_freqs * xi0 / xi
    # frequencies for wavelet centered in xi with Q wavelets per octave
    # freqs = xi0 + (xi_freqs - xi0) * Q

    num, den = b_function(freqs)
    num2, den2 = b_function(freqs / 2)
    numpi, denpi = b_function(freqs / 2 + 0.5)

    stable_den = np.empty_like(freqs)
    stable_den[freqs != 0] = np.sqrt(den[freqs != 0])  / (2 * np.pi * freqs[freqs != 0]) ** 4
    # protection in omega = 0
    stable_den[freqs == 0] = 2 ** (-4)


    mask = np.mod(freqs, 2) != 1
    stable_den[mask] *= np.sqrt(den2[mask] / denpi[mask])
    mask = np.mod(freqs, 2) == 1
    # protection in omega = 2pi [4pi]
    stable_den[mask] = np.sqrt(den2[mask]) / (np.pi * freqs[mask]) ** 4

    psi_hat = np.sqrt(numpi / (num * num2)) * stable_den
    psi_hat[freqs < 0] = 0

    return psi_hat

@jit
def battle_lemarie_phi(N, Q, xi_min):
    xi0 = BL_XI0  # mother wavelet center

    abs_freqs = np.fft.fftfreq(N)
    freqs = abs_freqs * xi0 / xi_min
    # freqs = xi_freqs * Q

    num, den = b_function(freqs)

    stable_den = np.empty_like(freqs)
    stable_den[freqs != 0] = np.sqrt(den[freqs != 0]) / (2 * np.pi * freqs[freqs != 0]) ** 4
    stable_den[freqs == 0] = 2 ** (-4)

    phi_hat = stable_den / np.sqrt(num)

    return phi_hat


@jit
def b_function(freqs, eps=1e-7):
    cos2 = np.cos(freqs * np.pi) ** 2
    sin2 = np.sin(freqs * np.pi) ** 2

    num = 5 + 30 * cos2 + 30 * sin2 * cos2 + 70 * cos2 ** 2 + 2 * sin2 ** 2 * cos2 + 2 * sin2 ** 3 / 3
    num /= 105 * 2 ** 8
    sin8 = sin2 ** 4

    return num, sin8



def compute_bump_steerable_parameters(N, Q, high_freq=0.5):
    return compute_battle_lemarie_parameters(N, Q, high_freq=high_freq)


def low_pass_constants(Q):
    """Function computing the ideal amplitude and variance for the low-pass of a bump
    wavelet dictionary, given the number of wavelets per scale Q.
    The amplitude and variance are computed by minimizing the frame error eta:
        1 - eta <= sum psi_la ** 2 <= 1 + eta
    Simple models are then fitted to compute those values quickly.
    The computation was done using gamma = 1.
    """
    ampl = -0.04809858889110362 + 1.3371665071917382 * np.sqrt(Q)
    xi2sigma = np.exp(-0.35365794431968484 - 0.3808886546835562 / Q)
    return ampl, xi2sigma


@jit
def bump_steerable_psi(N, Q, xi):
    abs_freqs = np.linspace(0, 1, N + 1)[:-1]
    psi = hwin((abs_freqs - xi) / xi, 1.)

    return psi


# @jit
# def bump_steerable_psi(N, Q, xi):
#     sigma = xi * BS_xi2sigma

#     abs_freqs = np.linspace(0, 1, N + 1)[:-1]
#     psi = hwin((abs_freqs - xi) / sigma, 1.)

#     return psi


@jit
def bump_steerable_phi(N, Q, xi_min):
    ampl, xi2sigma = low_pass_constants(Q)
    sigma = xi_min * xi2sigma

    abs_freqs = np.abs(np.fft.fftfreq(N))
    phi = ampl * np.exp(- (abs_freqs / (2 * sigma)) ** 2)

    return phi


# @jit
# def bump_steerable_phi(N, Q, xi_min):
#     sigma = xi_min * BS_xi2sigma

#     abs_freqs = np.abs(np.fft.fftfreq(N))
#     phi = hwin(abs_freqs / sigma, 1.)

#     return phi


@jit
def hwin(freqs, gamma1):
    psi_hat = np.zeros_like(freqs)
    idx = np.abs(freqs) < gamma1

    psi_hat[idx] = np.exp(1. / (freqs[idx] ** 2 - gamma1 ** 2))
    psi_hat *= np.exp(1 / gamma1 ** 2)

    return psi_hat


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    N = 5
    T = 2 ** (N + 2)
    Q = 3
    freqs = np.fft.fftfreq(T)
    fact = math.pow(2, 1 / Q)
    high_freq = 0.45

    # wavelet_type = "battle-lemarie"
    wavelet_type = "bump-steerable"
    # if wavelet_type == "battle-lemarie":
    #     print("Using center frequency {} for mother wavelet".format(BL_XI0))

    #     abs_freqs = np.linspace(0, 1, T + 1)[:-1]

    #     phi = battle_lemarie_phi(T, 1, fact ** (-N * Q)) * np.sqrt(Q)
    #     psi = [battle_lemarie_psi(T, 1, 0.45 * fact ** (-nq)) for nq in range(N * Q)]
    #     psi.append(phi)
    #     psi = np.stack(psi, axis=0)

    #     fxi = 0.34
    #     fpsi = battle_lemarie_psi(T, 1, fxi)
    #     mean = np.sum(abs_freqs * fpsi ** 2) / np.sum(fpsi ** 2)
    #     print("center frequency of {}: {}".format(fxi, mean))
    #     print("ratio: {}".format(mean / fxi))
    #     print("value at 0-: {:.2e}".format(psi[0, -1]))

    psi = [bump_steerable_psi(T, Q, high_freq * fact ** (-nq)) for nq in range(N * Q)]
    phi = bump_steerable_phi(T, Q, high_freq * fact ** (- N * Q + 1))
    psi = np.stack(psi + [phi], axis=0)
    # psi = np.stack(psi, axis=0)

    abs_freqs = np.linspace(0, 1, T + 1)[:-1]
    # plt.figure()
    # plt.plot(abs_freqs, psi[0, :])
    # plt.show()

    plt.figure()
    # for k in range(psi.shape[0]):
    #     plt.plot(abs_freqs, psi[k, :])
    plt.plot(abs_freqs, np.sum(psi ** 2, axis=0), color='r', linewidth=3)
    plt.show()
