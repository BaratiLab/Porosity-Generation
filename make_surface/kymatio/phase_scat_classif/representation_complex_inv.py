from utils import periodize, unpad, cdgmm, add_imaginary_part, prepare_padding_size, \
    pad, modulus, cast, periodic_dis, periodic_signed_dis
from complex_utils import conjugate, phase_exp, mul
from torch.autograd import Variable
from FFT import fft_c2c, ifft_c2r, ifft_c2c
import torch
import torch.nn.functional as F
from filters_banks import filters_bank, phase_filters_bank
import numpy as np
import torch
from tqdm import tqdm

def pad0(input):
    out_ = F.pad(input, (0,) * 4, mode='reflect').unsqueeze(input.dim())
    return torch.cat([out_, Variable(input.data.new(out_.size()).zero_())], 4)

#from representation_complex import phase_harmonic_cor
# phase_harmonic_cor corresponds to phase harmonic correlation interactions. In order to reduce the number of
# coefficients, the phase filters bank in A and A_prime can be different.
# Note: some issues may arise for odd Q due to rounding errors

#Problèmes de bord
#Replace mul by cdgmm?
#Rajouter avertissements sur l-max
def phase_harmonic_cor(input, phi, psi, J, L, delta, l_max):

    M, N = input.size(-2), input.size(-1)

    M_padded, N_padded = M, N # prepare_padding_size(M, N, J)

    nb_channels = (J * delta - (delta * (delta + 1)) // 2) * L * (2 * l_max + 1) + J * L * l_max

    S = Variable(input.data.new(input.size(0), input.size(1), nb_channels, M, N, 2)) # M_padded//(2**J)-2, N_padded//(2**J)-2, 2))
    
    # All convolutions are performed as products in Fourier. We first pad the input and compute its FFT
    print('input shape', input.shape)
    padded_input = pad0(input) # , J)
    print('padded_input shape', padded_input.shape)
    input_f = fft_c2c(padded_input)

    n = 0

    # We now compute the ReLU phase conditioned wavelet transform for each side
    for n_1 in range(len(psi)):
        j_1 = psi[n_1]['j']
        theta_1 = psi[n_1]['theta']

        W_f = cdgmm(input_f, psi[n_1][0])
        W_c = ifft_c2c(W_f)

        for n_2 in range(len(psi)):
            j_2 = psi[n_2]['j']
            theta_2 = psi[n_2]['theta']

            if (j_1 < j_2 <= j_1 + delta and periodic_dis(theta_1, theta_2, L) <= l_max) \
                    or (j_1 == j_2 and 0 <= periodic_signed_dis(theta_1, theta_2, L) <= l_max):

                W_f_prime = cdgmm(input_f, psi[n_2][0])
                W_c_prime = ifft_c2c(W_f_prime)
                W_exp_c_k_prime = conjugate(phase_exp(W_c_prime, 2**(j_2-j_1)))

                # We can then compute correlation coefficients
                W_exp_k_k_prime = mul(W_c, W_exp_c_k_prime) #cdgmm?
                #W_exp_k_k_prime_f = fft_c2c(W_exp_k_k_prime)

                #C_f = periodize(cdgmm(W_exp_k_k_prime_f, phi[0]), k=2**J)
                #C_c = ifft_c2c(C_f)
                
                S[..., n, :, :, :] = W_exp_k_k_prime # unpad(C_c, cplx=True)
                n = n + 1

    return S


# compute spatial averaging
def compute_phase_harmonic_cor_inv(X, J, L, delta, l_max, batch_size):

    M, N = X.shape[-2], X.shape[-1]
    M_padded, N_padded = M, N # prepare_padding_size(M, N, J)

    filters = filters_bank(M_padded, N_padded, J, L)

    psi = filters['psi']
    phi = [filters['phi'][j] for j in range(J)]

    psi, phi = cast(psi, phi, torch.cuda.FloatTensor)
    # cast in torch.FloatTensor to keep on CPU
    
    # renoamlize each psi to have max hat psi = 1, only at res=0
    for n_1 in range(len(psi)):
        j_1 = psi[n_1]['j']
        theta_1 = psi[n_1]['theta']
        print('max filter at j1=',j_1,',theta_1=',theta_1,' is ',psi[n_1][0].max())
        psi[n_1][0] = psi[n_1][0] / psi[n_1][0].max()
    
    phase_harmonics = phase_harmonic_cor(X[0:batch_size].cuda(), phi, psi, J, L, delta, l_max).cpu() # output (nb,nc,nch,M,N,2)
    # remove .cuda() to keep on CPU

    nb_batches = X.shape[0] // batch_size

    for idx_batch in tqdm(range(1, nb_batches)):
        phase_harmonics_tmp = phase_harmonic_cor(X[idx_batch * batch_size: (idx_batch + 1) * batch_size].cuda(), phi,
                                                 psi, J, L, delta, l_max).cpu()
        # remove .cuda() to keep on CPU
        phase_harmonics = torch.cat([phase_harmonics, phase_harmonics_tmp], dim=0)

    phase_harmonics_inv = torch.mean(torch.mean(phase_harmonics,-2,True),-3,True)

    return phase_harmonics_inv
