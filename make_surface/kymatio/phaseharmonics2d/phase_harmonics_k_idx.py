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
    Pad, unpad, SubInitMean, StablePhaseExp, PhaseExpSk, PhaseHarmonic, mul, conjugate
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
        check_for_nan = True
        #self.meta = None
        #self.modulus = Modulus()
        self.pad = Pad(0, pre_pad = self.pre_pad)
        #self.subsample_fourier = SubsampleFourier()
        #self.phaseexp = StablePhaseExp.apply
        #self.subinitmean = SubInitMean(2)
        #self.phase_exp = PhaseExpSk(keep_k_dim=True,check_for_nan=False)
        self.phase_harmonics = PhaseHarmonic(check_for_nan=check_for_nan)
        
        self.M_padded, self.N_padded = self.M, self.N
        filters = filter_bank(self.M_padded, self.N_padded, self.J, self.L, self.addhaar, self.cache)
     
        self.Psi = filters['psi']
        self.Phi = [filters['phi'][j] for j in range(self.J)]

        self.filt_tensor = self.filters_tensor()
        self.idx_wph = self.compute_idx()
        #print(self.idx_wph['la1'])
        #print(self.idx_wph['la2'])
        #print(self.idx_wph['k1'])
        #print(self.idx_wph['k2'])
        
    def filters_tensor(self):
        J = self.J
        L = self.L
        hatpsi = self.Psi
        filt = np.zeros((J, L, self.M, self.N), dtype=np.complex_) # TODO L->L2        

        for n_1 in range(len(hatpsi)):
            j_1 = hatpsi[n_1]['j']
            theta_1 = hatpsi[n_1]['theta']
            print('max hatpsi at j1=',j_1,',theta_1=',theta_1,' is ',hatpsi[n_1][0].max())
            hatpsi[n_1][0] = hatpsi[n_1][0] / hatpsi[n_1][0].max()
            print('hatpsi shape at res=0',hatpsi[n_1][0].shape)
            print('l2 norm hat psi at j1=',j_1,',theta_1=',theta_1,' is ',hatpsi[n_1][0].norm('fro'))
        
        for n in range(len(hatpsi)):
            j = hatpsi[n]['j']
            theta = hatpsi[n]['theta']
            filt[j, theta, :,:] = hatpsi[n][0][...,0].numpy() + 1j*hatpsi[n][0][...,1].numpy() # psi_signal
            #TODO filt[j, L+theta, :,:] = np.fft.fft2(np.conj(np.fft.ifft2(psi_signal)))

        filters = np.stack((np.real(filt), np.imag(filt)), axis=-1)
        return torch.FloatTensor(filters) # (J,L,M,N,2)

    def compute_idx(self):
        l_max=self.l_max
        L = self.L
        J = self.J
        j_max = self.j_max
      
        idx_la1 = []
        idx_la2 = []
        idx_k1 = []
        idx_k2 = []
        
        for j1 in range(J):
            for theta1 in range(L):
                k1 = 1
                for j2 in range(J):
                    for theta2 in range(L):
                        if (j1 < j2 <= j1 + j_max and periodic_dis(theta1, theta2, L) <= l_max) \
                           or (j1 == j2 and 0 <= periodic_signed_dis(theta1, theta2, L) <= l_max):
                            k2 = 2**(j2-j1)
                            idx_la1.append(L*j1+theta1)
                            idx_la2.append(L*j2+theta2)
                            idx_k1.append(k1)
                            idx_k2.append(k2)
        
        idx_wph = dict()
        idx_wph['la1'] = torch.tensor(idx_la1).type(torch.long)
        idx_wph['k1'] = torch.tensor(idx_k1).type(torch.long).float().unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        idx_wph['la2'] = torch.tensor(idx_la2).type(torch.long)
        idx_wph['k2'] = torch.tensor(idx_k2).type(torch.long).float().unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        
        return idx_wph 
        
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

#        hatphi = self.Phi # low pass
#        hatpsi = self.Psi # high pass
        
        pad = self.pad
        #modulus = self.modulus
        
        # denote
        # nb=batch number
        # nc=number of color channels
        # input: (nb,nc,M,N)
        x_c = pad(input) # add zeros to imag part -> (nb,nc,M,N,2)
        hatx_c = fft2_c2c(x_c) # fft2 -> (nb,nc,M,N,2)
        
        nb = hatx_c.shape[0]
        nc = hatx_c.shape[1]
        nb_channels = self.idx_wph['la1'].shape[0]
        print('nbchannels',nb_channels)
        Sout = input.new(nb, nc, nb_channels, \
                         1, 1, 2) # no spatial phiJ # (nb,nc,nb_channels,1,1,2)
        
        hatpsi_la = self.filt_tensor # (J,L,M,N,2)
     
        for idxb in range(nb):
            for idxc in range(nc):
                hatx_bc = hatx_c[idxb,idxc,:,:,:] # (M,N,2)
                hatxpsi_bc = cdgmm(hatpsi_la, hatx_bc) # (J,L,M,N,2)
                #print( 'hatxpsi_bc shape', hatxpsi_bc.shape )
                xpsi_bc = ifft2_c2c(hatxpsi_bc)
                # reshape to (1,J*L,M,N,2)
                xpsi_bc = xpsi_bc.view(1,J*L,M,N,2)
                # select la1, et la2, P = |la1| 
                xpsi_bc_la1 = torch.index_select(xpsi_bc, 1, self.idx_wph['la1']) # (1,P,M,N,2)
                xpsi_bc_la2 = torch.index_select(xpsi_bc, 1, self.idx_wph['la2']) # (1,P,M,N,2)
                print('xpsi la1 shape', xpsi_bc_la1.shape)
                print('xpsi la2 shape', xpsi_bc_la2.shape)
                k1 = self.idx_wph['k1']
                k2 = self.idx_wph['k2']
                xpsi_bc_la1k1 = self.phase_harmonics(xpsi_bc_la1, k1) # (1,P,M,N,2)
                xpsi_bc_la2k2 = self.phase_harmonics(xpsi_bc_la2, -k2) # (1,P,M,N,2)
                # compute mean spatial
                corr_xpsi_bc = mul(xpsi_bc_la1k1,xpsi_bc_la2k2)
                corr_bc = torch.mean(torch.mean(corr_xpsi_bc,-2,True),-3,True) # (1,P,1,1,2)
                Sout[idxb,idxc,:,:,:,:] = corr_bc[0,:,:,:,:]
                
        return Sout

    def __call__(self, input):
        return self.forward(input)
