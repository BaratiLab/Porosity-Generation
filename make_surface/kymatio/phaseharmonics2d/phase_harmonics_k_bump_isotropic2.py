# assume process is both isotropic and has plain symmetry
# removed last chunkid to make it more compact
# isotropic case, implement basic phase harmonics
# remove negitive freq. in q

__all__ = ['PhaseHarmonics2d']

import warnings
import math
import torch
import numpy as np
import scipy.io as sio
from .backend import cdgmm, Modulus, fft, \
    Pad, SubInitSpatialMeanC, SubInitMeanIso, PhaseHarmonicsIso, mulcu, conjugate
from .filter_bank import filter_bank
from .utils import fft2_c2c, ifft2_c2c, periodic_dis

class PhaseHarmonics2d(object):
    def __init__(self, M, N, J, L, delta_j, delta_l, delta_k, nb_chunks, chunk_id):
        self.M, self.N, self.J, self.L = M, N, J, L # size of image, max scale, number of angles [0,pi]
        self.dj = delta_j # max scale interactions
        self.dl = delta_l # max angular interactions
        self.dk = delta_k #
        self.K = 2**self.dj + self.dk + 1
        self.k = torch.arange(0, self.K).type(torch.float) # vector between [0,..,K-1]
        self.nb_chunks = nb_chunks # number of chunks to cut whp cov
        self.chunk_id = chunk_id
        assert( self.chunk_id < self.nb_chunks ) # chunk_id = 0..nb_chunks-1, are the wph cov
        if self.dl != self.L:
            raise (ValueError('delta_l must be = L'))

        self.pre_pad = False # no padding
        self.cache = False # cache filter bank
        self.build()

    def build(self):
        check_for_nan = False # True
        self.modulus = Modulus()
        self.pad = Pad(0, pre_pad = self.pre_pad)
        self.phase_harmonics = PhaseHarmonicsIso.apply
        self.M_padded, self.N_padded = self.M, self.N
        self.filters_tensor()
        if self.chunk_id < self.nb_chunks:
            self.idx_wph = self.compute_idx()
            self.this_wph = self.get_this_chunk(self.nb_chunks, self.chunk_id)
            self.subinitmean = SubInitMeanIso()
            self.subinitmeanJ = SubInitSpatialMeanC()

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
        hatpsi = np.stack((np.real(fftpsi), np.imag(fftpsi)), axis=-1)

        self.hatpsi = torch.FloatTensor(hatpsi) # (J,L2,M,N,2)
        self.hatphi = torch.FloatTensor(hatphi) # (M,N,2)


    def get_this_chunk(self, nb_chunks, chunk_id):
        # cut self.idx_wph into smaller pieces
        nb_cov = len(self.idx_wph['la1'])
        max_chunk = nb_cov // nb_chunks
        nb_cov_chunk = np.zeros(nb_chunks,dtype=np.int32)
        for idxc in range(nb_chunks):
            if idxc < nb_chunks-1:
                nb_cov_chunk[idxc] = int(max_chunk)
            else:
                nb_cov_chunk[idxc] = int(nb_cov - max_chunk*(nb_chunks-1))
                assert(nb_cov_chunk[idxc] > 0)

        this_wph = dict()
        offset = int(0)
        for idxc in range(nb_chunks):
            if idxc == chunk_id:
                this_wph['la1'] = self.idx_wph['la1'][offset:offset+nb_cov_chunk[idxc]]
                this_wph['la2'] = self.idx_wph['la2'][offset:offset+nb_cov_chunk[idxc]]      
            offset = offset + nb_cov_chunk[idxc]

        print('this chunk', chunk_id, ' size is ', len(this_wph['la1']), ' among ', nb_cov)

        return this_wph

    def compute_idx(self):
        L = self.L
        L2 = L*2
        Q = L2
        J = self.J
        dj = self.dj
        dl = self.dl
        dk = self.dk
        K = self.K

        idx_la1 = []
        idx_la2 = []

        # j1=j2, k1=1, k2=0 or 1
        for j1 in range(J):
            for q1 in range(L+1): # from q=0..L
                k1 = 1
                j2 = j1
                q2 = q1
                k2 = 0
                idx_la1.append(K*Q*j1 + K*q1 + k1)
                idx_la2.append(K*Q*j2 + K*q2 + k2)
                k2 = 1
                idx_la1.append(K*Q*j1 + K*q1 + k1)
                idx_la2.append(K*Q*j2 + K*q2 + k2)

        # k1 = 0
        # k2 = 0
        # j1 = j2
        for j1 in range(J):
            for q1 in range(L+1):
                k1 = 0
                j2 = j1
                q2 = q1
                k2 = 0
                idx_la1.append(K*Q*j1 + K*q1 + k1)
                idx_la2.append(K*Q*j2 + K*q2 + k2)
        
        # k1 = 0
        # k2 = 0,1,2
        # j1+1 <= j2 <= min(j1+dj,J-1)
        for j1 in range(J):
            for q1 in range(L+1):
                k1 = 0
                for j2 in range(j1+1, min(j1+dj+1, J)):
                    q2 = q1
                    for k2 in range(3):
                        idx_la1.append(K*Q*j1 + K*q1 + k1)
                        idx_la2.append(K*Q*j2 + K*q2 + k2)
        
        # k1 = 1
        # k2 = 2^(j2-j1)±dk
        # j1+1 <= j2 <= min(j1+dj,J-1)
        for j1 in range(J):
            for q1 in range(L+1):
                k1 = 1
                q2 = q1
                for j2 in range(j1+1, min(j1+dj+1, J)):
                    for k2 in range(max(0, 2**(j2-j1)-dk), 2**(j2-j1)+dk+1):
                        idx_la1.append(K*Q*j1 + K*q1 + k1)
                        idx_la2.append(K*Q*j2 + K*q2 + k2)

        idx_wph = dict()
        idx_wph['la1'] = torch.tensor(idx_la1).type(torch.long)
        idx_wph['la2'] = torch.tensor(idx_la2).type(torch.long)

        return idx_wph

    def _type(self, _type, devid=None):
        self.hatpsi = self.hatpsi.type(_type)
        self.hatphi = self.hatphi.type(_type)
        if devid is not None:
            self.hatpsi = self.hatpsi.to(devid)
            self.hatphi = self.hatphi.to(devid)
        self.pad.padding_module.type(_type)
        self.k = self.k.type(_type)
        return self

    def cuda(self, devid=0):
        """
            Moves tensors to the GPU
        """
        print('call cuda')
        if self.chunk_id < self.nb_chunks:
            self.this_wph['la1'] = self.this_wph['la1'].type(torch.cuda.LongTensor).to(devid)
            self.this_wph['la2'] = self.this_wph['la2'].type(torch.cuda.LongTensor).to(devid)

        return self._type(torch.cuda.FloatTensor, devid)

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
        Q = L2
        
        dj = self.dj
        dl = self.dl
        dk = self.dk
        K = self.K
        k = self.k
        pad = self.pad

        # denote
        # nb=batch number
        # nc=number of color channels
        # input: (nb,nc,M,N)
        x_c = pad(input) # add zeros to imag part -> (nb,nc,M,N,2)
        hatx_c = fft2_c2c(x_c) # fft2 -> (nb,nc,M,N,2)
        if self.chunk_id < self.nb_chunks:
            nb = hatx_c.shape[0]
            nc = hatx_c.shape[1]
            hatpsi_la = self.hatpsi # (J,L2,M,N,2)
            assert(nb==1 and nc==1) # for submeanC
            nb_channels = self.this_wph['la1'].shape[0]
            if self.chunk_id < self.nb_chunks-1:
                Sout = input.new(nb, nc, nb_channels, 1, 1, 1)
            else:
                Sout = input.new(nb, nc, nb_channels+1, 1, 1, 1)
            idxb = 0
            idxc = 0
            hatx_bc = hatx_c[idxb,idxc,:,:,:] # (M,N,2)
            hatxpsi_bc = cdgmm(hatpsi_la, hatx_bc) # (J,L2,M,N,2)
            xpsi_bc = ifft2_c2c(hatxpsi_bc)
            xpsi_bc = xpsi_bc.unsqueeze(-2)  # (J,L2,M,N,1,2)
            xpsi_ph_bc = self.phase_harmonics(xpsi_bc, k)  # (J,L2,M,N,K,2)
            # permute to put angle 2nd to last
            xpsi_ph_bc_ = xpsi_ph_bc.permute(0,4,2,3,1,5) # (J,K,M,N,L2,2)
            # fft in angles
            xpsi_iso_bc = torch.fft(xpsi_ph_bc_, 1, normalized=True)  # (J,K,M,N,Q,2)
            # permute again
            xpsi_iso_bc_ = xpsi_iso_bc.permute(0,4,1,2,3,5)  # (J,Q,K,M,N,2)
            # sub spatial mean for Q = 0
            xpsi_iso_bc0 = self.subinitmean(xpsi_iso_bc_)
            # reshape to (1,J*L,M,N,2)
            xpsi_iso_bc0_ = xpsi_iso_bc0.view(1,J*Q*K,M,N,2)
            
            # select la1, et la2, P_c = number of |la1| in this chunk
            xpsi_bc_la1 = torch.index_select(xpsi_iso_bc0_, 1, self.this_wph['la1']) # (1,P_c,M,N,2)
            xpsi_bc_la2 = torch.index_select(xpsi_iso_bc0_, 1, self.this_wph['la2']) # (1,P_c,M,N,2)
            
            # compute mean spatial
            corr_xpsi_bc = mulcu(xpsi_bc_la1, conjugate(xpsi_bc_la2)) # (1,P_c,M,N,2)
            corr_bc = torch.mean(torch.mean(corr_xpsi_bc,-2,True),-3,True) # (1,P_c,1,1,2)
            Sout[idxb,idxc,0:nb_channels,0,0,0] = corr_bc[0,:,0,0,0] # only keep real part
            
            if self.chunk_id == self.nb_chunks-1:
                # ADD 1 chennel for spatial phiJ
                # add l2 phiJ to last channel
                hatxphi_c = cdgmm(hatx_c, self.hatphi) # (nb,nc,M,N,2)
                xphi_c = ifft2_c2c(hatxphi_c)
                # submean from spatial M N
                xphi0_c = self.subinitmeanJ(xphi_c)
                xphi0_mod = self.modulus(xphi0_c) # (nb,nc,M,N,2)
                xphi0_mod2 = mulcu(xphi0_mod,xphi0_mod) # (nb,nc,M,N,2)
                xphi0_mean = torch.mean(torch.mean(xphi0_mod2,-2,True),-3,True)
                Sout[idxb,idxc,-1,0,0,0] = xphi0_mean[idxb,idxc,0,0,0]
                
        return Sout

    def __call__(self, input):
        return self.forward(input)
