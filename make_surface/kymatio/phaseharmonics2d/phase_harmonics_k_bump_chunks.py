# implement basic phase harmonics
# based on John's code to check correctness
# Case: phase_harmonic_cor in representation_complex
#      do not create new Sout for each forward

__all__ = ['PhaseHarmonics2d']

import warnings
import math
import torch
import numpy as np
import scipy.io as sio
import torch.nn.functional as F
from .backend import cdgmm, Modulus, SubsampleFourier, fft, \
    Pad, unpad, SubInitSpatialMeanC, StablePhaseExp, PhaseExpSk, PhaseHarmonic, mul, conjugate
from .filter_bank import filter_bank
from .utils import compute_padding, fft2_c2c, ifft2_c2r, ifft2_c2c, periodic_dis, periodic_signed_dis

class PhaseHarmonics2d(object):
    def __init__(self, M, N, J, L, delta_j, delta_l, delta_k, max_chunk):
        self.M, self.N, self.J, self.L = M, N, J, L # size of image, max scale, number of angles [0,pi]
        self.dj = delta_j # max scale interactions
        self.dl = delta_l # max angular interactions
        self.dk = delta_k #
        self.max_chunk = max_chunk # max size of each chunk 
        if self.dl > self.L:
            raise (ValueError('delta_l must be <= L'))

        self.pre_pad = False # no padding
        self.cache = False # cache filter bank
        self.build()

    def build(self):
        check_for_nan = True
        #self.meta = None
        self.modulus = Modulus()
        self.pad = Pad(0, pre_pad = self.pre_pad)
        #self.subsample_fourier = SubsampleFourier()
        #self.phaseexp = StablePhaseExp.apply

        #self.phase_exp = PhaseExpSk(keep_k_dim=True,check_for_nan=False)
        self.phase_harmonics = PhaseHarmonic(check_for_nan=check_for_nan)

        self.M_padded, self.N_padded = self.M, self.N

        #filters = filter_bank(self.M_padded, self.N_padded, self.J, self.L, False, self.cache) # no Haar

        #self.Psi = filters['psi']
        #self.Phi = [filters['phi'][j] for j in range(self.J)]
        self.filters_tensor()
        self.idx_wph = self.compute_idx()
        self.idx_wph_chunks, self.n_chunks, self.chunk_size = self.balanced_chunks()
        
        self.subinitmean1 = []
        self.subinitmean2 = []
        for idxc in range(self.n_chunks):
            self.subinitmean1.append(SubInitSpatialMeanC())
            self.subinitmean2.append(SubInitSpatialMeanC())
        
        self.subinitmeanJ = SubInitSpatialMeanC()
        
        #print(self.idx_wph['la1'])
        #print(self.idx_wph['la2'])
        #print(self.idx_wph['k1'])
        #print(self.idx_wph['k2'])

    def filters_tensor(self):
        # TODO load bump steerable wavelets
        J = self.J
        L = self.L
        L2 = L*2

        assert(self.M == self.N)
        matfilters = sio.loadmat('./matlab/filters/bumpsteerableg1_fft2d_N' + str(self.N) + '_J' + str(self.J) + '_L' + str(self.L) + '.mat')

        fftphi = matfilters['filt_fftphi'].astype(np.complex_)
        hatphi = np.stack((np.real(fftphi), np.imag(fftphi)), axis=-1)

        fftpsi = matfilters['filt_fftpsi'].astype(np.complex_)
        #print(self.hatpsi.dtype)
        hatpsi = np.stack((np.real(fftpsi), np.imag(fftpsi)), axis=-1)

        self.hatpsi = torch.FloatTensor(hatpsi) # (J,L2,M,N,2)
        self.hatphi = torch.FloatTensor(hatphi) # (M,N,2)

        #print('filter shapes')
        #print(self.hatpsi.shape)
        #print(self.hatphi.shape)

    def balanced_chunks(self):
        # cut self.idx_wph into smaller pieces
        #print('la1 shape',self.idx_wph['la1'].shape)
        
        nb_cov = len(self.idx_wph['la1'])
        print('nb cov is', nb_cov)
        
        n_chunks = math.ceil(nb_cov/self.max_chunk)
        nb_cov_chunk = np.zeros(n_chunks,dtype=np.int32)
        for idxc in range(n_chunks):
            if idxc < n_chunks-1:
                nb_cov_chunk[idxc] = int(self.max_chunk)
            else:
                nb_cov_chunk[idxc] = int(nb_cov - self.max_chunk*(n_chunks-1))
                assert(nb_cov_chunk[idxc] > 0)

        print('nb cov chunk is', nb_cov_chunk)

        idx_wph_chunks = dict()
        offset = int(0)
        for idxc in range(n_chunks):
            idx_wph_chunks[('la1',idxc)] = self.idx_wph['la1'][offset:offset+nb_cov_chunk[idxc]]
            idx_wph_chunks[('la2',idxc)] = self.idx_wph['la2'][offset:offset+nb_cov_chunk[idxc]]
            idx_wph_chunks[('k1',idxc)] = self.idx_wph['k1'][:,offset:offset+nb_cov_chunk[idxc],:,:]
            idx_wph_chunks[('k2',idxc)] = self.idx_wph['k2'][:,offset:offset+nb_cov_chunk[idxc],:,:]
            offset = offset + nb_cov_chunk[idxc]
            #print(self.idx_wph_chunks[('k1',idxc)])
        #assert(False)
        return idx_wph_chunks, n_chunks, nb_cov_chunk
    
    def compute_idx(self):
        L = self.L
        L2 = L*2
        J = self.J
        dj = self.dj
        dl = self.dl
        dk = self.dk

        idx_la1 = []
        idx_la2 = []
        idx_k1 = []
        idx_k2 = []

        # TODO add dl
        # j1=j2, k1=1, k2=0 or 1
        for j1 in range(J):
            for ell1 in range(L2):
                k1 = 1
                j2 = j1
                for ell2 in range(L2):
                    if periodic_dis(ell1, ell2, L2) <= dl:
                        k2 = 0
                        idx_la1.append(L2*j1+ell1)
                        idx_la2.append(L2*j2+ell2)
                        idx_k1.append(k1)
                        idx_k2.append(k2)
                        k2 = 1
                        idx_la1.append(L2*j1+ell1)
                        idx_la2.append(L2*j2+ell2)
                        idx_k1.append(k1)
                        idx_k2.append(k2)

        # k1 = 0
        # k2 = 0,1,2
        # j1+1 <= j2 <= min(j1+dj,J-1)
        for j1 in range(J):
            for ell1 in range(L2):
                k1 = 0
                for j2 in range(j1+1,min(j1+dj+1,J)):
                    for ell2 in range(L2):
                        if periodic_dis(ell1, ell2, L2) <= dl:
                            for k2 in range(3):
                                idx_la1.append(L2*j1+ell1)
                                idx_la2.append(L2*j2+ell2)
                                idx_k1.append(k1)
                                idx_k2.append(k2)

        # k1 = 1
        # k2 = 2^(j2-j1)±dk
        # j1+1 <= j2 <= min(j1+dj,J-1)
        for j1 in range(J):
            for ell1 in range(L2):
                k1 = 1
                for ell2 in range(L2):
                    if periodic_dis(ell1, ell2, L2) <= dl:
                        for j2 in range(j1+1,min(j1+dj+1,J)):
                            for k2 in range(max(0,2**(j2-j1)-dk,2**(j2-j1)+dk+1)):
                                idx_la1.append(L2*j1+ell1)
                                idx_la2.append(L2*j2+ell2)
                                idx_k1.append(k1)
                                idx_k2.append(k2)

        idx_wph = dict()
        idx_wph['la1'] = torch.tensor(idx_la1).type(torch.long)
        idx_wph['k1'] = torch.tensor(idx_k1).type(torch.long).float().unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        idx_wph['la2'] = torch.tensor(idx_la2).type(torch.long)
        idx_wph['k2'] = torch.tensor(idx_k2).type(torch.long).float().unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

        return idx_wph

    def _type(self, _type):
        self.hatpsi = self.hatpsi.type(_type)
        self.hatphi = self.hatphi.type(_type)
        #print('in _type',type(self.hatpsi))
        self.pad.padding_module.type(_type)
        return self
    
    def cuda(self):
        """
            Moves tensors to the GPU
        """
        print('call cuda')
        self.idx_wph['la1'] = self.idx_wph['la1'].type(torch.cuda.LongTensor)
        self.idx_wph['la2'] = self.idx_wph['la2'].type(torch.cuda.LongTensor)
        self.idx_wph['k1'] = self.idx_wph['k1'].type(torch.cuda.FloatTensor)
        self.idx_wph['k2'] = self.idx_wph['k2'].type(torch.cuda.FloatTensor)
        for idxc in range(self.n_chunks):
            self.idx_wph_chunks[('la1',idxc)] = self.idx_wph_chunks[('la1',idxc)].type(torch.cuda.LongTensor)
            self.idx_wph_chunks[('la2',idxc)] = self.idx_wph_chunks[('la2',idxc)].type(torch.cuda.LongTensor)
            self.idx_wph_chunks[('k1',idxc)] = self.idx_wph_chunks[('k1',idxc)].type(torch.cuda.FloatTensor)
            self.idx_wph_chunks[('k2',idxc)] = self.idx_wph_chunks[('k2',idxc)].type(torch.cuda.FloatTensor)            
        
        return self._type(torch.cuda.FloatTensor)

    def cpu(self):
        """
            Moves tensors to the CPU
        """
        print('call cpu')
        return self._type(torch.FloatTensor)

    def forward(self, input):
        J = self.J
        M = self.M
        N = self.N
        L2 = self.L*2
        dj = self.dj
        dl = self.dl

#        hatphi = self.Phi # low pass
#        hatpsi = self.Psi # high pass

        pad = self.pad
        #phi = self.Phi
        #modulus = self.modulus

        # denote
        # nb=batch number
        # nc=number of color channels
        # input: (nb,nc,M,N)
        x_c = pad(input) # add zeros to imag part -> (nb,nc,M,N,2)
        hatx_c = fft2_c2c(x_c) # fft2 -> (nb,nc,M,N,2)

        nb = hatx_c.shape[0]
        nc = hatx_c.shape[1]
        nb_channels = self.idx_wph['la1'].shape[0] + 1
        # ADD 1 chennel for spatial phiJ
        #print('nbchannels',nb_channels)
        Sout = input.new(nb, nc, nb_channels, \
                         1, 1, 2) # (nb,nc,nb_channels,1,1,2)

        hatpsi_la = self.hatpsi # (J,L2,M,N,2)
        assert(nb==1 and nc==1) # for submeanC
        for idxb in range(nb):
            for idxc in range(nc):
                hatx_bc = hatx_c[idxb,idxc,:,:,:] # (M,N,2)
 #               print('hatpsi_la is cuda?',hatpsi_la.is_cuda)
#                print('hatx_bc is cuda?',hatx_bc.is_cuda)
                hatxpsi_bc = cdgmm(hatpsi_la, hatx_bc) # (J,L2,M,N,2)
                #print( 'hatxpsi_bc shape', hatxpsi_bc.shape )
                xpsi_bc = ifft2_c2c(hatxpsi_bc)
                # reshape to (1,J*L,M,N,2)
                xpsi_bc = xpsi_bc.view(1,J*L2,M,N,2)
                offset = 0
                for idx_chunk in range(self.n_chunks):
                    # select la1, et la2, P_c = number of |la1| in this chunk
                    xpsi_bc_la1 = torch.index_select(xpsi_bc, 1, self.idx_wph_chunks[('la1',idx_chunk)]) # (1,P_c,M,N,2)
                    xpsi_bc_la2 = torch.index_select(xpsi_bc, 1, self.idx_wph_chunks[('la2',idx_chunk)]) # (1,P_c,M,N,2)
                    #print('xpsi la1 shape', xpsi_bc_la1.shape)
                    #print('xpsi la2 shape', xpsi_bc_la2.shape)
                    k1 = self.idx_wph_chunks[('k1',idx_chunk)]
                    k2 = self.idx_wph_chunks[('k2',idx_chunk)]
                    xpsi_bc_la1k1 = self.phase_harmonics(xpsi_bc_la1, k1) # (1,P_c,M,N,2)
                    xpsi_bc_la2k2 = self.phase_harmonics(xpsi_bc_la2, -k2) # (1,P_c,M,N,2)
                    # sub spatial mean along M and N
                    xpsi0_bc_la1k1 = self.subinitmean1[idx_chunk](xpsi_bc_la1k1) # (1,P_c,M,N,2)
                    xpsi0_bc_la2k2 = self.subinitmean2[idx_chunk](xpsi_bc_la2k2) # (1,P_c,M,N,2)
                    # compute mean spatial
                    corr_xpsi_bc = mul(xpsi0_bc_la1k1,xpsi0_bc_la2k2) # (1,P_c,M,N,2)
                    corr_bc = torch.mean(torch.mean(corr_xpsi_bc,-2,True),-3,True) # (1,P_c,1,1,2)
                    Sout[idxb,idxc,offset:offset+self.chunk_size[idx_chunk],:,:,:] = corr_bc[0,:,:,:,:]
                    offset = offset + self.chunk_size[idx_chunk]
                    
        # add l2 phiJ to last channel
        hatxphi_c = cdgmm(hatx_c, self.hatphi) # (nb,nc,M,N,2)
        xpsi_c = ifft2_c2c(hatxphi_c)
        # submean from spatial M N
        xpsi0_c = self.subinitmeanJ(xpsi_c)
        xpsi0_mod = self.modulus(xpsi0_c) # (nb,nc,M,N,2)
        xpsi0_mod2 = mul(xpsi0_mod,xpsi0_mod) # (nb,nc,M,N,2)
        Sout[:,:,nb_channels-1,:,:,:] = torch.mean(torch.mean(xpsi0_mod2,-2,True),-3,True)

        return Sout

    def __call__(self, input):
        return self.forward(input)
