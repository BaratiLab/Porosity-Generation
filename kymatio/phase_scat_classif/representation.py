from utils import periodize, unpad, cdgmm, add_imaginary_part, prepare_padding_size, \
    pad, modulus, cast, periodic_dis, periodic_signed_dis
from torch.autograd import Variable
from FFT import fft_c2c, ifft_c2r, ifft_c2c
import torch.nn.functional as F
from filters_banks import filters_bank, phase_filters_bank
import numpy as np
import torch
from tqdm import tqdm

# Note: we compute here the representation using CUDA GPU acceleration. In order to compute it on CPU, input tensors
# need to remain on CPU and thus .cuda() needs to be removed and filters banks needs to be casted in torch.FloatTensor
# instead of torch.cuda.FloatTensor when computing the blocks in the differnet compute_ functions (see at bottom)


# We compute separately the different blocks of our representation
# We start with translation scattering of order 2
def scattering(input, phi, psi_1, psi_2, J, L_1, L_2, Q, square=False):

    M, N = input.size(-2), input.size(-1)
    M_padded, N_padded = prepare_padding_size(M, N, J)

    nb_channels = 1 + J * Q * L_1 + (J * (J - 1)) // 2 * L_1 * L_2 + (Q - 1) * ((J - 1) * (J - 2)) // 2 * L_1 * L_2

    S = Variable(input.data.new(input.size(0), input.size(1), nb_channels, M_padded//(2**J)-2, N_padded//(2**J)-2))

    # All convolutions are performed as products in Fourier. We first pad the input and compute its FFT
    U_0 = pad(input, J)
    U_0_c = fft_c2c(U_0)

    # We can then compute scattering of order 0 by filtering with phi (local average of the input)
    S_0_c = periodize(cdgmm(U_0_c, phi[0]), k=2**J)
    S_0_r = ifft_c2r(S_0_c)

    n = 0
    S[..., n, :, :] = unpad(S_0_r)
    n = n + 1

    # We compute order 1 by first computing the modulus wavelet transform with appropriate downsampling
    for n_1 in range(len(psi_1)):
        j_1 = psi_1[n_1]['j']
        U_1_c = cdgmm(U_0_c, psi_1[n_1][0])
        if np.floor(j_1) > 0:
            U_1_c = periodize(U_1_c, k=2**int(np.floor(j_1)))
        U_1_c = fft_c2c(modulus(ifft_c2c(U_1_c)))

        # We can now compute scattering of order 1 by filtering with phi
        S_1_c = periodize(cdgmm(U_1_c, phi[int(np.floor(j_1))]), k=2**(J-int(np.floor(j_1))))
        S_1_r = ifft_c2r(S_1_c)
        S[..., n, :, :] = unpad(S_1_r)
        n = n + 1

        # We cascade wavelet modulus operator to compute order 2
        for n_2 in range(len(psi_2)):
            j_2 = psi_2[n_2]['j']
            if j_1 + 1 <= j_2:  # we just want one wavelet per octave for the second order
                U_2_c = periodize(cdgmm(U_1_c, psi_2[n_2][int(np.floor(j_1))]), k=2**(j_2-int(np.floor(j_1))))
                if square:
                    U_2_c = fft_c2c(modulus(ifft_c2c(U_2_c))**2)
                else:
                    U_2_c = fft_c2c(modulus(ifft_c2c(U_2_c)))

                # We can now compute scattering of order 2 by filtering with phi
                S_2_c = periodize(cdgmm(U_2_c, phi[j_2]), k=2**(J-j_2))
                S_2_r = ifft_c2r(S_2_c)

                S[..., n, :, :] = unpad(S_2_r)
                n = n + 1
    return S


# phase_harmonic_cor corresponds to phase harmonic correlation interactions. In order to reduce the number of
# coefficients, the phase filters bank in A and A_prime can be different.
def phase_harmonic_cor(input, phi, psi_phase_A, psi_phase_A_prime, J, L, Q, A, A_prime, delta, l_max):

    M, N = input.size(-2), input.size(-1)

    M_padded, N_padded = prepare_padding_size(M, N, J)

    nb_channels = (J * Q * delta - Q * (delta * (delta + 1)) // 2) * L * min(L, (2 * l_max + 1)) * A * A_prime + \
                  J * Q * min(L*(L+1)//2, L*(l_max + 1)) * A * A_prime

    S = Variable(input.data.new(input.size(0), input.size(1), nb_channels, M_padded//(2**J)-2, N_padded//(2**J)-2))

    # All convolutions are performed as products in Fourier. We first pad the input and compute its FFT
    padded_input = pad(input, J)
    input_f = fft_c2c(padded_input)

    n = 0

    # We now compute the ReLU phase conditioned wavelet transform for each side
    for n_1 in range(len(psi_phase_A)):
        j_1 = psi_phase_A[n_1]['j']
        theta_1 = psi_phase_A[n_1]['theta']

        W_f = cdgmm(input_f, psi_phase_A[n_1][0])
        W_r = ifft_c2r(W_f)  # We take here the real part, hence ifft_c2r
        U_1_r = F.relu(W_r)

        for n_2 in range(len(psi_phase_A_prime)):
            j_2 = psi_phase_A_prime[n_2]['j']
            theta_2 = psi_phase_A_prime[n_2]['theta']

            if ((j_1 < j_2 <= j_1 + delta and ((j_2-j_1) % 1 == 0)) and periodic_dis(theta_1, theta_2, L) <=
                min(L//2, l_max)) or (j_1 == j_2 and ((l_max >= L//2 and theta_2 >= theta_1) or
                                 (l_max < L//2 and periodic_signed_dis(theta_1, theta_2, L) <= l_max))):

                W_f_prime = cdgmm(input_f, psi_phase_A_prime[n_2][0])
                W_r_prime = ifft_c2r(W_f_prime)
                U_1_r_prime = F.relu(W_r_prime)

                # We can then compute correlation coefficients
                U_1_r_U_1_r_prime = U_1_r * U_1_r_prime
                U_1_r_U_1_r_prime_c = add_imaginary_part(U_1_r_U_1_r_prime)
                U_1_r_U_1_r_prime_f = fft_c2c(U_1_r_U_1_r_prime_c)

                S_f = periodize(cdgmm(U_1_r_U_1_r_prime_f, phi[0]), k=2**J)
                S_r = ifft_c2r(S_f)

                S[..., n, :, :] = unpad(S_r)
                n = n + 1

    return S


# phase harmonic color channels interactions
def phase_harmonic_cor_color(input, phi, psi_phase_A, psi_phase_A_prime, J, L, Q, A, A_prime, delta, l_max):

    M, N = input.size(-2), input.size(-1)

    M_padded, N_padded = prepare_padding_size(M, N, J)

    nb_channels = (J * Q * delta - Q * (delta * (delta + 1)) // 2) * L * min(L, (2 * l_max + 1)) * A * A_prime + \
                  J * Q * min(L * (L + 1) // 2, L * (l_max + 1)) * A * A_prime

    nb_channels_colors_input = input.size(1)
    nb_channels_colors_inter = (nb_channels_colors_input*(nb_channels_colors_input-1))//2

    S = Variable(input.data.new(input.size(0), nb_channels_colors_inter, nb_channels,
                                M_padded//(2**J)-2, N_padded//(2**J)-2))

    # All convolutions are performed as products in Fourier. We first pad the input and compute its FFT
    padded_input = pad(input, J)
    input_f = fft_c2c(padded_input)

    n = 0

    # We now compute the ReLU phase conditioned wavelet transform for each side
    for n_1 in range(len(psi_phase_A)):
        j_1 = psi_phase_A[n_1]['j']
        theta_1 = psi_phase_A[n_1]['theta']

        W_f = cdgmm(input_f, psi_phase_A[n_1][0])
        W_r = ifft_c2r(W_f)  # We take here the real part, hence ifft_c2r
        U_1_r = F.relu(W_r)

        for n_2 in range(len(psi_phase_A_prime)):
            j_2 = psi_phase_A_prime[n_2]['j']
            theta_2 = psi_phase_A_prime[n_2]['theta']

            if ((j_1 < j_2 <= j_1 + delta and (j_2-j_1) % 1 == 0) and periodic_dis(theta_1, theta_2, L) <=
                min(L // 2, l_max)) or (j_1 == j_2 and ((l_max >= L // 2 and theta_2 >= theta_1) or
                                     (l_max < L // 2 and periodic_signed_dis(theta_1, theta_2, L) <= l_max))):

                W_f_prime = cdgmm(input_f, psi_phase_A_prime[n_2][0])
                W_r_prime = ifft_c2r(W_f_prime)
                U_1_r_prime = F.relu(W_r_prime)

                n_ci = 0
                for n_c1 in range(nb_channels_colors_input):
                    for n_c2 in range(n_c1 + 1, nb_channels_colors_input):

                        # We can then compute correlation coefficients
                        U_1_r_U_1_r_prime_c1_c2 = U_1_r[:, n_c1, ...] * U_1_r_prime[:, n_c2, ...]
                        U_1_r_U_1_r_prime_c1_c2 = U_1_r_U_1_r_prime_c1_c2.unsqueeze(1)
                        U_1_r_U_1_r_prime_c1_c2_c = add_imaginary_part(U_1_r_U_1_r_prime_c1_c2)
                        U_1_r_U_1_r_prime_c1_c2_f = fft_c2c(U_1_r_U_1_r_prime_c1_c2_c)

                        S_f = periodize(cdgmm(U_1_r_U_1_r_prime_c1_c2_f, phi[0]), k=2**J)
                        S_r = ifft_c2r(S_f)
                        S_r = S_r.squeeze()

                        S[:, n_ci, n, :, :] = unpad(S_r)
                        n_ci = n_ci + 1
                n = n + 1

    return S


# phase_harmonic_cor_2nd_order corresponds to phase harmonic correlation interactions of second order. In order to
# reduce the number of coefficients, the phase filters bank in A and A_prime can be different.
def phase_harmonic_cor_2nd_order(input, phi, psi, psi_phase_A, psi_phase_A_prime, J, L, L_phase, Q, A, A_prime, l_max):

    M, N = input.size(-2), input.size(-1)

    M_padded, N_padded = prepare_padding_size(M, N, J)

    nb_channels = A * A_prime * L * L_phase * min(L_phase, (2 * l_max + 1)) * (J * Q * (J * Q - 1) * (J * Q - 2)) // 6

    S = Variable(input.data.new(input.size(0), input.size(1), nb_channels, M_padded//(2**J)-2, N_padded//(2**J)-2))

    # All convolutions are performed as products in Fourier. We first pad the input and compute its FFT
    padded_input = pad(input, J)
    input_f = fft_c2c(padded_input)

    n = 0

    # We now compute the ReLU phase conditioned wavelet transform for each side
    for n_1 in range(len(psi)):
        j_1 = psi[n_1]['j']

        U_1_c = cdgmm(input_f, psi[n_1][0])
        #if np.floor(j_1) > 0:
        #    U_1_c = periodize(U_1_c, k=2 ** int(np.floor(j_1)))
        U_1_c = fft_c2c(modulus(ifft_c2c(U_1_c)))


        for n_2 in range(len(psi_phase_A)):
            j_2 = psi_phase_A[n_2]['j']
            theta_2 = psi_phase_A[n_2]['theta']

            if j_1 < j_2:

                W_f = cdgmm(U_1_c, psi_phase_A[n_2][0])
                W_r = ifft_c2r(W_f)
                U_2_r = F.relu(W_r)

                for n_3 in range(len(psi_phase_A_prime)):
                    j_3 = psi_phase_A_prime[n_3]['j']
                    theta_3 = psi_phase_A_prime[n_3]['theta']

                    if j_2 < j_3 and periodic_dis(theta_2, theta_3, L_phase) <= min(L//2, l_max):

                        W_f_prime = cdgmm(U_1_c, psi_phase_A_prime[n_3][0])
                        W_r_prime = ifft_c2r(W_f_prime)
                        U_2_r_prime = F.relu(W_r_prime)

                        # We can then compute correlation coefficients
                        U_2_r_U_2_r_prime = U_2_r * U_2_r_prime
                        U_2_r_U_2_r_prime_c = add_imaginary_part(U_2_r_U_2_r_prime)
                        U_2_r_U_2_r_prime_f = fft_c2c(U_2_r_U_2_r_prime_c)

                        S_f = periodize(cdgmm(U_2_r_U_2_r_prime_f, phi[0]), k=2**J)
                        S_r = ifft_c2r(S_f)

                        S[..., n, :, :] = unpad(S_r)
                        n = n + 1

    return S


# modulus_cor corresponds to wavelet modulus correlations at different scales and angles
def modulus_cor(input, phi, psi, J, L, Q, delta):

    M, N = input.size(-2), input.size(-1)

    M_padded, N_padded = prepare_padding_size(M, N, J)

    nb_channels = (J * Q * delta - (delta * (delta + 1)) // 2) * L ** 2 + J * Q * (L * (L + 1)) // 2

    S = Variable(input.data.new(input.size(0), input.size(1), nb_channels, M_padded//(2**J)-2, N_padded//(2**J)-2))

    # All convolutions are performed as products in Fourier. We first pad the input and compute its FFT
    padded_input = pad(input, J)
    input_f = fft_c2c(padded_input)

    n = 0

    for n_1 in range(len(psi)):
        j_1 = psi[n_1]['j']
        theta_1 = psi[n_1]['theta']

        W_f = cdgmm(input_f, psi[n_1][0])
        U_1_r = modulus(ifft_c2c(W_f))

        for n_2 in range(len(psi)):
            j_2 = psi[n_2]['j']
            theta_2 = psi[n_2]['theta']

            if (j_1 < j_2 <= j_1 + delta/Q) or (j_1 == j_2 and theta_2 >= theta_1):
                W_f_prime = cdgmm(input_f, psi[n_2][0])
                U_1_r_prime = modulus(ifft_c2c(W_f_prime))

                U_1_r_U_1_r_prime = U_1_r * U_1_r_prime
                U_1_r_U_1_r_prime_f = fft_c2c(U_1_r_U_1_r_prime)

                S_f = periodize(cdgmm(U_1_r_U_1_r_prime_f, phi[0]), k=2**J)
                S_r = ifft_c2r(S_f)

                S[..., n, :, :] = unpad(S_r)
                n = n + 1

    return S


# wavelet modulus color channels correlations
def modulus_cor_color(input, phi, psi, J, L, Q, delta):
    M, N = input.size(-2), input.size(-1)

    M_padded, N_padded = prepare_padding_size(M, N, J)

    nb_channels = (J * Q * delta - (delta * (delta + 1)) // 2) * L ** 2 + J * Q * (L * (L + 1)) // 2

    nb_channels_colors_input = input.size(1)
    nb_channels_colors_inter = (nb_channels_colors_input * (nb_channels_colors_input - 1)) // 2

    S = Variable(input.data.new(input.size(0), nb_channels_colors_inter, nb_channels,
                                M_padded // (2 ** J) - 2, N_padded // (2 ** J) - 2))

    # All convolutions are performed as products in Fourier. We first pad the input and compute its FFT
    padded_input = pad(input, J)
    input_f = fft_c2c(padded_input)

    n = 0

    # We now compute the ReLU phase conditioned wavelet transform for each side
    for n_1 in range(len(psi)):
        j_1 = psi[n_1]['j']
        theta_1 = psi[n_1]['theta']

        W_f = cdgmm(input_f, psi[n_1][0])
        U_1_r = modulus(ifft_c2c(W_f))

        for n_2 in range(len(psi)):
            j_2 = psi[n_2]['j']
            theta_2 = psi[n_2]['theta']

            if (j_1 < j_2 <= j_1 + delta/Q) or (j_1 == j_2 and theta_2 >= theta_1):
                W_f_prime = cdgmm(input_f, psi[n_2][0])
                U_1_r_prime = modulus(ifft_c2c(W_f_prime))

                n_ci = 0
                for n_c1 in range(nb_channels_colors_input):
                    for n_c2 in range(n_c1 + 1, nb_channels_colors_input):
                        # We can then compute correlation coefficients
                        U_1_r_U_1_r_prime_c1_c2 = U_1_r[:, n_c1, ...] * U_1_r_prime[:, n_c2, ...]
                        U_1_r_U_1_r_prime_c1_c2 = U_1_r_U_1_r_prime_c1_c2.unsqueeze(1)
                        U_1_r_U_1_r_prime_c1_c2_f = fft_c2c(U_1_r_U_1_r_prime_c1_c2)

                        S_f = periodize(cdgmm(U_1_r_U_1_r_prime_c1_c2_f, phi[0]), k=2 ** J)
                        S_r = ifft_c2r(S_f)
                        S_r = S_r.squeeze()

                        S[:, n_ci, n, :, :] = unpad(S_r)
                        n_ci = n_ci + 1
                n = n + 1

    return S


# We now compute the representations for each block

# Remove .cuda() to keep inputs on CPU and cast filters banks in torch.FloatTensor
# instead of torch.cuda.FloatTensor to perform calculations on CPU in each compute_

def compute_scat(X, J, L_1, L_2, Q, batch_size, square=False):

    M, N = X.shape[-2], X.shape[-1]
    M_padded, N_padded = prepare_padding_size(M, N, J)

    filters_1 = filters_bank(M_padded, N_padded, J, L_1, Q)

    psi_1 = filters_1['psi']
    phi = [filters_1['phi'][j] for j in range(J)]

    psi_1, phi = cast(psi_1, phi, torch.cuda.FloatTensor)
    # cast in torch.FloatTensor to keep on CPU

    filters_2 = filters_bank(M_padded, N_padded, J, L_2)  # Q = 1

    psi_2 = filters_2['psi']
    phi = [filters_2['phi'][j] for j in range(J)]

    psi_2, phi = cast(psi_2, phi, torch.cuda.FloatTensor)
    # cast in torch.FloatTensor to keep on CPU

    scat = scattering(X[0:batch_size].cuda(), phi, psi_1, psi_2, J, L_1, L_2, Q, square).cpu().numpy()
    # remove .cuda() to keep on CPU

    nb_batches = X.shape[0] // batch_size

    for idx_batch in tqdm(range(1, nb_batches)):
        scat_tmp = scattering(X[idx_batch * batch_size: (idx_batch + 1) * batch_size].cuda(), phi, psi_1, psi_2, J,
                               L_1, L_2, Q, square).cpu().numpy()
        # remove .cuda() to keep on CPU
        scat = np.concatenate([scat, scat_tmp], axis=0)

    scat = scat.reshape(X.shape[0], -1)

    return scat


    # To align conventions with scattering, we assume L designates the sampling in angles on [0, pi]. Hence, we need a
    # filters bank with 2 * L angles since as we take alphas on [0, pi], we need to take thetas in [0, 2pi]
def compute_phase_harmonic_cor(X, J, L, Q, A, A_prime, delta, l_max, batch_size):

    M, N = X.shape[-2], X.shape[-1]
    M_padded, N_padded = prepare_padding_size(M, N, J)

    filters_phase_A = phase_filters_bank(M_padded, N_padded, J, L, A, Q)

    psi_phase_A = filters_phase_A['psi']
    phi = [filters_phase_A['phi'][0]]
    psi_phase_A, phi = cast(psi_phase_A, phi, torch.cuda.FloatTensor)
    # cast in torch.FloatTensor to keep on CPU

    filters_phase_A_prime = phase_filters_bank(M_padded, N_padded, J, L, A_prime, Q)

    psi_phase_A_prime = filters_phase_A_prime['psi']
    psi_phase_A_prime, phi = cast(psi_phase_A_prime, phi, torch.cuda.FloatTensor)
    # cast in torch.FloatTensor to keep on CPU

    phase_harmonics = phase_harmonic_cor(X[0:batch_size].cuda(), phi, psi_phase_A, psi_phase_A_prime, J, L, Q, A,
                                         A_prime, delta, l_max).cpu().numpy()
    # remove .cuda() to keep on CPU

    nb_batches = X.shape[0] // batch_size

    for idx_batch in tqdm(range(1, nb_batches)):
        phase_harmonics_tmp = phase_harmonic_cor(X[idx_batch * batch_size: (idx_batch + 1) * batch_size].cuda(), phi,
                                                 psi_phase_A, psi_phase_A_prime, J, L, Q, A, A_prime, delta,
                                                 l_max).cpu().numpy()
        # remove .cuda() to keep on CPU
        phase_harmonics = np.concatenate([phase_harmonics, phase_harmonics_tmp], axis=0)

    phase_harmonics = phase_harmonics.reshape(X.shape[0], -1)

    return phase_harmonics


    # To align conventions with scattering, we assume L designates the sampling in angles on [0, pi]. Hence, we need a
    # filters bank with 2 * L angles since as we take alphas on [0, pi], we need to take thetas in [0, 2pi]
def compute_phase_harmonic_color_cor(X, J, L, Q, A, A_prime, delta, l_max, batch_size):

    M, N = X.shape[-2], X.shape[-1]
    M_padded, N_padded = prepare_padding_size(M, N, J)

    filters_phase_A = phase_filters_bank(M_padded, N_padded, J, L, A, Q)

    psi_phase_A = filters_phase_A['psi']
    phi = [filters_phase_A['phi'][0]]
    psi_phase_A, phi = cast(psi_phase_A, phi, torch.cuda.FloatTensor)
    # cast in torch.FloatTensor to keep on CPU

    filters_phase_A_prime = phase_filters_bank(M_padded, N_padded, J, L, A_prime, Q)

    psi_phase_A_prime = filters_phase_A_prime['psi']
    psi_phase_A_prime, phi = cast(psi_phase_A_prime, phi, torch.cuda.FloatTensor)
    # cast in torch.FloatTensor to keep on CPU

    phase_harmonics_color = phase_harmonic_cor_color(X[0:batch_size].cuda(), phi, psi_phase_A, psi_phase_A_prime, J,
                                                     L, Q, A, A_prime, delta, l_max).cpu().numpy()
    # remove .cuda() to keep on CPU

    nb_batches = X.shape[0] // batch_size

    for idx_batch in tqdm(range(1, nb_batches)):
        phase_harmonics_color_tmp = phase_harmonic_cor_color(X[idx_batch * batch_size: (idx_batch + 1) * batch_size].
                                                             cuda(), phi, psi_phase_A, psi_phase_A_prime, J, L, Q, A,
                                                             A_prime, delta, l_max).cpu().numpy()
        # remove .cuda() to keep on CPU
        phase_harmonics_color = np.concatenate([phase_harmonics_color, phase_harmonics_color_tmp], axis=0)

    phase_harmonics_color = phase_harmonics_color.reshape(X.shape[0], -1)

    return phase_harmonics_color


def compute_phase_harmonic_2nd_order(X, J, L, L_phase, Q, A, A_prime, l_max, batch_size):

    M, N = X.shape[-2], X.shape[-1]
    M_padded, N_padded = prepare_padding_size(M, N, J)

    filters = filters_bank(M_padded, N_padded, J, L, Q)
    psi = filters['psi']
    phi = [filters['phi'][0]]
    psi, phi = cast(psi, phi, torch.cuda.FloatTensor)
    # cast in torch.FloatTensor to keep on CPU

    filters_phase_A = phase_filters_bank(M_padded, N_padded, J, L_phase, A, Q)

    psi_phase_A = filters_phase_A['psi']
    phi = [filters_phase_A['phi'][0]]
    psi_phase_A, phi = cast(psi_phase_A, phi, torch.cuda.FloatTensor)
    # cast in torch.FloatTensor to keep on CPU

    filters_phase_A_prime = phase_filters_bank(M_padded, N_padded, J, L_phase, A_prime, Q)

    psi_phase_A_prime = filters_phase_A_prime['psi']
    psi_phase_A_prime, phi = cast(psi_phase_A_prime, phi, torch.cuda.FloatTensor)
    # cast in torch.FloatTensor to keep on CPU

    phase_harmonics_2nd_order = phase_harmonic_cor_2nd_order(X[0:batch_size].cuda(), phi, psi, psi_phase_A,
                                                             psi_phase_A_prime, J, L, L_phase, Q, A, A_prime,
                                                             l_max).cpu().numpy()
    # remove .cuda() to keep on CPU

    nb_batches = X.shape[0] // batch_size

    for idx_batch in tqdm(range(1, nb_batches)):
        phase_harmonics_2nd_order_tmp = phase_harmonic_cor_2nd_order(X[idx_batch * batch_size: (idx_batch + 1) *
                                                                 batch_size].cuda(), phi, psi, psi_phase_A,
                                                                 psi_phase_A_prime, J, L, L_phase, Q, A, A_prime,
                                                                 l_max).cpu().numpy()
        # remove .cuda() to keep on CPU
        phase_harmonics_2nd_order = np.concatenate([phase_harmonics_2nd_order, phase_harmonics_2nd_order_tmp], axis=0)

    phase_harmonics_2nd_order = phase_harmonics_2nd_order.reshape(X.shape[0], -1)

    return phase_harmonics_2nd_order


def compute_modulus_cor(X, J, L, Q, delta, batch_size):

    M, N = X.shape[-2], X.shape[-1]
    M_padded, N_padded = prepare_padding_size(M, N, J)

    filters = filters_bank(M_padded, N_padded, J, L, Q)

    psi = filters['psi']
    phi = [filters['phi'][j] for j in range(J)]

    psi, phi = cast(psi, phi, torch.cuda.FloatTensor)
    # cast in torch.FloatTensor to keep on CPU

    modulus_coeff = modulus_cor(X[0:batch_size].cuda(), phi, psi, J, L, Q, delta).cpu().numpy()
    # remove .cuda() to keep on CPU

    nb_batches = X.shape[0] // batch_size

    for idx_batch in tqdm(range(1, nb_batches)):
        modulus_coeff_tmp = modulus_cor(X[idx_batch * batch_size: (idx_batch+1) * batch_size].cuda(), phi, psi, J, L, Q,
                                        delta).cpu().numpy()
        # remove .cuda() to keep on CPU
        modulus_coeff = np.concatenate([modulus_coeff, modulus_coeff_tmp], axis=0)

    modulus_coeff = modulus_coeff.reshape(X.shape[0], -1)

    return modulus_coeff


def compute_modulus_cor_color(X, J, L, Q, delta, batch_size):

    M, N = X.shape[-2], X.shape[-1]
    M_padded, N_padded = prepare_padding_size(M, N, J)

    filters = filters_bank(M_padded, N_padded, J, L, Q)

    psi = filters['psi']
    phi = [filters['phi'][j] for j in range(J)]

    psi, phi = cast(psi, phi, torch.cuda.FloatTensor)
    # cast in torch.FloatTensor to keep on CPU

    modulus_color_coeff = modulus_cor_color(X[0:batch_size].cuda(), phi, psi, J, L, Q, delta).cpu().numpy()
    # remove .cuda() to keep on CPU

    nb_batches = X.shape[0] // batch_size

    for idx_batch in tqdm(range(1, nb_batches)):
        modulus_color_coeff_tmp = modulus_cor_color(X[idx_batch * batch_size: (idx_batch+1) * batch_size].cuda(), phi,
                                                    psi, J, L, Q, delta).cpu().numpy()
        # remove .cuda() to keep on CPU
        modulus_color_coeff = np.concatenate([modulus_color_coeff, modulus_color_coeff_tmp], axis=0)

    modulus_color_coeff = modulus_color_coeff.reshape(X.shape[0], -1)

    return modulus_color_coeff