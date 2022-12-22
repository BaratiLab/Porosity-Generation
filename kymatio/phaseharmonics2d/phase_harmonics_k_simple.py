# implement basic phase harmonics
# based on John's code to check correctness
# Case: phase_harmonic_cor in representation_complex
# TODO need extend L toward 2L
#      comptue cov rather than corr
#      do not create new Sout for each forward

__all__ = ['PhaseHarmonics2d']

import warnings
import torch
import numpy as np
import torch.nn.functional as F
from .backend import cdgmm, Modulus, SubsampleFourier, fft, \
    Pad, unpad, SubInitMean, StablePhaseExp, PhaseExpSk, mul, conjugate
from .filter_bank import filter_bank
from .utils import compute_padding, fft2_c2c, ifft2_c2r, ifft2_c2c, periodic_dis, periodic_signed_dis 

class PhaseHarmonics2d(object):
    def __init__(self, M, N, J, L, j_max, l_max, addhaar=False, gpu=False):
        self.M, self.N, self.J, self.L = M, N, J, L # size of image, max scale, number of angles [0,pi]
        self.j_max = j_max # max scale interactions
        self.l_max = l_max # max angular interactions
        self.addhaar = addhaar # filter bank with haar filters
        self.gpu = gpu # if to use gpu
        if self.l_max > self.L:
            raise (ValueError('l_max must be <= L'))
        
        self.pre_pad = False # no padding
        self.cache = False # cache filter bank
        self.build()
        
    def build(self):
        self.meta = None
        self.modulus = Modulus()
        self.pad = Pad(0, pre_pad = self.pre_pad)
        self.subsample_fourier = SubsampleFourier()
        #self.phaseexp = StablePhaseExp.apply
        self.subinitmean = SubInitMean(2)
        self.phase_exp = PhaseExpSk(keep_k_dim=True,check_for_nan=False)
        self.M_padded, self.N_padded = self.M, self.N
        filters = filter_bank(self.M_padded, self.N_padded, self.J, self.L, self.addhaar, self.cache)           
        self.Psi = filters['psi']
        self.Phi = [filters['phi'][j] for j in range(self.J)]
        if self.addhaar:
            self.Psi0 = filters['psi0']
    
    def cuda(self):
        """
            Moves tensors to the GPU
        """
        return self._type(torch.cuda.FloatTensor)

    def cpu(self):
        """
            Moves tensors to the CPU
        """
        return self._type(torch.FloatTensor)

    def forward(self, input):
        J = self.J
        M = self.M
        N = self.N
        L = self.L
        j_max = self.j_max
        l_max = self.l_max

        hatphi = self.Phi # low pass
        hatpsi = self.Psi # high pass
        
        pad = self.pad
        modulus = self.modulus
        
        # denote
        # nb=batch number
        # nc=number of color channels
        # input: (nb,nc,M,N)
        x_c = pad(input) # add zeros to imag part -> (nb,nc,M,N,2)
        hatx_c = fft2_c2c(x_c) # fft2 -> (nb,nc,M,N,2)
        
        set_meta = False
        if self.meta is None:
            set_meta = True
            self.meta = dict()
            # renoamlize to each psi to have max hat psi = 1, only res=0
            for n_1 in range(len(hatpsi)):
                j_1 = hatpsi[n_1]['j']
                theta_1 = hatpsi[n_1]['theta']
                print('max hatpsi at j1=',j_1,',theta_1=',theta_1,' is ',hatpsi[n_1][0].max())
                hatpsi[n_1][0] = hatpsi[n_1][0] / hatpsi[n_1][0].max()
                print('hatpsi shape at res=0',hatpsi[n_1][0].shape)
                print('l2 norm hat psi at j1=',j_1,',theta_1=',theta_1,' is ',hatpsi[n_1][0].norm('fro'))
        
        # out coefficients:
        nb_channels = (J * j_max - (j_max * (j_max + 1)) // 2) * L * (2 * l_max + 1) + J * L * l_max
        Sout = input.data.new(input.size(0), input.size(1), nb_channels, \
                              1, 1, 2) # no spatial phiJ # (nb,nc,nb_channels,1,1,2)
        
        idxc = 0 # channel index
        for n1 in range(len(hatpsi)):
            # compute x * psi_{j1,theta1}, no subsampling
            j1 = hatpsi[n1]['j']
            theta1 = hatpsi[n1]['theta']
            k1 = 1
            hatxpsi_c = cdgmm(hatx_c, hatpsi[n1][0]) # (nb,nc,M,N,2)
            xpsi_c = ifft2_c2c(hatxpsi_c) # (nb,nc,M,N,2)
            
            for n2 in range(len(hatpsi)):
                j2 = hatpsi[n2]['j']
                theta2 = hatpsi[n2]['theta']
                
                if (j1 < j2 <= j1 + j_max and periodic_dis(theta1, theta2, L) <= l_max) \
                   or (j1 == j2 and 0 <= periodic_signed_dis(theta1, theta2, L) <= l_max):
                    k2 = 2**(j2-j1)
                    # compute [x * psi_{j2,thate2}]^{2^(j2-j1)}
                    hatxpsi_prime_c = cdgmm(hatx_c, hatpsi[n2][0]) # (nb,nc,M,N,2)
                    xpsi_prime_c = ifft2_c2c(hatxpsi_prime_c) # (nb,nc,M,N,2)
                    pexpsi_prime_c = conjugate(self.phase_exp(xpsi_prime_c,k2)) # (nb,nc,M,N,2)
                    
                    # We can then compute correlation coefficients
                    pemul_c = mul(xpsi_c, pexpsi_prime_c) # (nb,nc,M,N,2)
                    pecorr_c = torch.mean(torch.mean(pemul_c,-2,True),-3,True) # (nb,nc,1,1,2) #cdgmm vs. mul?
                    #print('idx,pecorr_c',idxc,pecorr_c.shape)
                    # compute mean along spatial domain, save to Sout
                    Sout[...,idxc,:,:,:] = pecorr_c
                    
                    if set_meta:
                        self.meta[idxc] = (j1,theta1,k1,j2,theta2,k2)
                    idxc = idxc + 1
        
        return Sout, self.meta

    def __call__(self, input):
        return self.forward(input)
