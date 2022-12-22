__all__ = ['Scattering']

import warnings
import torch
import numpy as np
import torch.nn.functional as F
from .backend import cdgmm, Modulus, SubsampleFourier, fft, Pad, unpad,\
 SubInitMean, StablePhaseExp, PhaseExp_par, mul, conjugate

from .filter_bank import filter_bank
from .utils import compute_padding, fft2_c2c, ifft2_c2r, ifft2_c2c, periodic_dis, periodic_signed_dis

class PhaseHarmonics(object):

    def __init__(self, M, N, J, K, L, delta, l_max, gpu=False,
                 k_type='log2', addhaar=False, order2=False):
        self.M, self.N, self.J, self.L = M, N, J, L
        self.pre_pad = False # no padding
        self.order2 = order2
        self.addhaar = addhaar
        self.cache = False
        self.K = K
        self.delta = delta
        self.l_max = l_max
        self.gpu = gpu
        if self.l_max > self.L:
            raise (
                ValueError('l_max must be <= L'))
        self.build(k_type = k_type)
        self.filters_tensor()
        self.phase_harm_cor_idx()


    def build(self, k_type='log'):
        self.modulus = Modulus()
        #self.pad = Pad(2**self.J, pre_pad = self.pre_pad)
        self.pad = Pad(0, pre_pad = self.pre_pad)
        self.subsample_fourier = SubsampleFourier()
        #self.phaseexp = StablePhaseExp.apply
        self.subinitmean = SubInitMean(2)
        self.phase_exp = PhaseExp_par(self.K, k_type=k_type, keep_k_dim=True,
                                  check_for_nan=False)
        # Create the filters
        #self.M_padded, self.N_padded = compute_padding(self.M, self.N, self.J)
        self.M_padded, self.N_padded = self.M, self.N
        filters = filter_bank(self.M_padded, self.N_padded, self.J, (self.L), self.addhaar, self.cache)
        #filters = filter_bank(self.M, self.N, self.J, self.L)
        self.Psi = filters['psi']

        self.Phi = [filters['phi'][j] for j in range(self.J)]
        if self.addhaar:
            self.Psi0 = filters['psi0']


    def filters_tensor(self):
        J = self.J
        L = self.L
        psi = self.Psi


        #filt = np.zeros((J, 2*L, self.M, self.N), dtype=np.complex_)
        filt = np.zeros((J, L, self.M, self.N), dtype=np.complex_)
        

        for n in range(len(psi)):
            j = psi[n]['j']
            theta = psi[n]['theta']
            psi_signal = psi[n][0][...,0].numpy() + 1j*psi[n][0][...,1].numpy()
            filt[j, theta, :,:] = psi_signal
            #filt[j, L+theta, :,:] = np.fft.fft2(np.conj(np.fft.ifft2(psi_signal)))


        filters = np.stack((np.real(filt), np.imag(filt)), axis=-1)

        self.filt_tensor = torch.FloatTensor(filters)


    def phase_harm_cor_idx(self):

        l_max=self.l_max
        L = self.L
        J = self.J
        delta = self.delta
        K = self.K

        idx1 = []
        idx2 = []
        

        for j1 in range(J):
            for theta1 in range(L):
                for j2 in range(J):
                    for theta2 in range(L):
                        if (j1 < j2 <= j1 + delta and periodic_dis(theta1, theta2, L) <= l_max) \
                                or (j1 == j2 and 0 <= periodic_signed_dis(theta1, theta2, L) <= l_max) :
                                    idx1.append(K*L*j1+L+theta1)
                                    idx2.append(K*L*j2+L*(j2-j1+1)+theta2)
        self.idx_phase_harm = (torch.tensor(idx1).type(torch.long),
                               torch.tensor(idx2).type(torch.long))

        



    def _type(self, _type):
#        for key, item in enumerate(self.Psi):
#            for key2, item2 in self.Psi[key].items():
#                if torch.is_tensor(item2):
#                    self.Psi[key][key2] = item2.type(_type)
        self.Phi = [v.type(_type) for v in self.Phi]
        self.pad.padding_module.type(_type)
        if self.addhaar:
            for key, item in self.Psi0.items():
                self.Psi0[key] = item.type(_type)

        self.filt_tensor = self.filt_tensor.type(_type)
        return self

    def cuda(self):
        """
            Moves the parameters of the scattering to the GPU
        """
        return self._type(torch.cuda.FloatTensor)

    def cpu(self):
        """
            Moves the parameters of the scattering to the CPU
        """
        return self._type(torch.FloatTensor)

    def forward(self, input):
        J = self.J
        M = self.M
        N = self.N
        phi = self.Phi
        psi = self.Psi
        n = 0
        modulus = self.modulus
        pad = self.pad
        idx1, idx2 = self.idx_phase_harm

        filt_tensor = self.filt_tensor

        filt_tensor = filt_tensor.unsqueeze(2).unsqueeze(2)

        U_r = pad(input)
        U_0_c = fft2_c2c(U_r)
        U_1_c = U_0_c.unsqueeze(0).unsqueeze(0).expand_as(filt_tensor)

        #U_1_c = U_1_c.expand_as(filt_tensor)

        conv = U_1_c.new(U_1_c.size())
        conv[..., 0] = U_1_c[..., 0]*filt_tensor[..., 0] - U_1_c[..., 1]*filt_tensor[..., 1]
        conv[..., 1] = U_1_c[..., 0]*filt_tensor[..., 1] + U_1_c[..., 1]*filt_tensor[..., 0]

        conv = ifft2_c2c(conv)

        harm = self.phase_exp(conv)
        #print(harm.size())
        harm = harm.view(-1, 1, 1, M, N, 2)

        corr_1 = torch.index_select(harm, 0, idx2)
        corr_2 = torch.index_select(harm, 0, idx1)

        corr = mul(corr_1, conjugate(corr_2))

        P = corr.mean(-2).mean(-2)

        return P


    def __call__(self, input):
        return self.forward(input)
