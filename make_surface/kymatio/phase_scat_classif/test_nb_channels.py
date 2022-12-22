from utils import periodize, unpad, cdgmm, add_imaginary_part, prepare_padding_size, \
    pad, modulus, periodic_dis, periodic_signed_dis, cast
import torch
from FFT import fft_c2c, ifft_c2r, ifft_c2c
from torch.autograd import Variable
from filters_banks import filters_bank, phase_filters_bank
import numpy as np


J=3
L=8
delta = 0
l_max = 0


M = 32
N = 32

M_padded, N_padded = prepare_padding_size(M, N, J)
nb_channels_colors_input = 3

filters = filters_bank(M_padded, N_padded, J, L)

psi = filters['psi']
phi = [filters['phi'][j] for j in range(J)]

psi, phi = cast(psi, phi, torch.cuda.FloatTensor)
# cast in torch.FloatTensor to keep on CPU

nb_channels = (J * (delta + 1) - (delta * (delta + 1)) // 2) * L * (2 * l_max + 1)



n = 0

# We now compute the ReLU phase conditioned wavelet transform for each side
for n_1 in range(len(psi)):
    j_1 = psi[n_1]['j']
    theta_1 = psi[n_1]['theta']

    for n_2 in range(len(psi)):
        j_2 = psi[n_2]['j']
        theta_2 = psi[n_2]['theta']

        if (j_1 <= j_2 <= j_1 + delta and periodic_dis(theta_1, theta_2, L) <= l_max):

            n_ci = 0
            for n_c1 in range(nb_channels_colors_input):
                for n_c2 in range(n_c1 + 1, nb_channels_colors_input):
                    n_ci = n_ci + 1
            n = n + 1

print(nb_channels)
print(n)
print(nb_channels*3*4*4)




