# cross scale interaction correlations 
# delta_l controls angular correlations
# delta_j controls scale correlations
__all__ = ['PhaseHarmonics2d']

import warnings
import math
import torch
import numpy as np
import scipy.io as sio
from .backend import cdgmm, Modulus, fft, \
    Pad, SubInitSpatialMeanC, PhaseHarmonics2, mulcu, conjugate
from .filter_bank import filter_bank
from .utils import fft2_c2c, ifft2_c2c, periodic_dis

class WaveletCovIsoScaleInter2d(object):
    def __init__(self, M, N, J, L, delta_j, nb_chunks, chunk_id, devid=0, filname='bumpsteerableg', filid=1):
        self.M, self.N, self.J, self.L = M, N, J, L # size of image, max scale, number of angles [0,pi]
        self.dj = delta_j # max scale interactions
        self.nb_chunks = nb_chunks # number of chunks to cut whp cov
        self.chunk_id = chunk_id
        self.devid = devid # gpu id
        self.filname = filname
        self.filid = filid
        self.haspsi0 = False
        assert( self.chunk_id <= self.nb_chunks )
        # chunk_id = 0..nb_chunks-1, are the cov of wavelets, chunk_id=nb_chunks is cov of phiJ
        self.pre_pad = False # no padding
        self.cache = False # cache filter bank
        self.build()
    
    def build(self):
        check_for_nan = False
        self.modulus = Modulus()
        self.pad = Pad(0, pre_pad = self.pre_pad)
        self.phase_harmonics = PhaseHarmonics2.apply
        self.M_padded, self.N_padded = self.M, self.N
        self.filters_tensor()
        if self.chunk_id < self.nb_chunks:
            self.idx_wph = self.compute_idx()
            self.this_wph = self.get_this_chunk(self.nb_chunks, self.chunk_id)
        else:
            self.subinitmeanJ = SubInitSpatialMeanC()
            
    def preselect_filters(self):
        assert(false)
    
    def filters_tensor(self):
        J = self.J
        L = self.L
        L2 = L*2
        
        assert(self.M == self.N)
        filpath = './matlab/filters/' + self.filname + str(self.filid) + '_fft2d_N'\
                  + str(self.N) + '_J' + str(self.J) + '_L' + str(self.L) + '.mat'
        matfilters = sio.loadmat(filpath)
        print('filter loaded:', filpath)
        
        fftphi = matfilters['filt_fftphi'].astype(np.complex_)
        hatphi = np.stack((np.real(fftphi), np.imag(fftphi)), axis=-1)
        
        if 'filt_fftpsi0' in matfilters:
            fftpsi0 = matfilters['filt_fftpsi0'].astype(np.complex_)
            hatpsi0 = np.stack((np.real(fftpsi0), np.imag(fftpsi0)), axis=-1)
            self.hatpsi0 = torch.FloatTensor(hatpsi0) # (M,N,2)
            self.haspsi0 = True
            print('compute psi0')
        
        fftpsi = matfilters['filt_fftpsi'].astype(np.complex_)
        hatpsi = np.stack((np.real(fftpsi), np.imag(fftpsi)), axis=-1)
        
        self.hatpsi = torch.FloatTensor(hatpsi) # (J,L2,M,N,2)
        self.hatphi = torch.FloatTensor(hatphi) # (M,N,2)
    
    def get_this_chunk(self, nb_chunks, chunk_id):
        # cut self.idx_wph into smaller pieces
        #print('la1 shape',self.idx_wph['la1'].shape)
        nb_cov = len(self.idx_wph['j1'])
        #print('nb cov is', nb_cov)
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
                this_wph['j1'] = self.idx_wph['j1'][offset:offset+nb_cov_chunk[idxc]]
                this_wph['j2'] = self.idx_wph['j2'][offset:offset+nb_cov_chunk[idxc]]
            offset = offset + nb_cov_chunk[idxc]
        
        print('this chunk', chunk_id, ' size is ', len(this_wph['j1']), ' among ', nb_cov)

        return this_wph
    
    def compute_idx(self):
        L = self.L
        L2 = L*2
        J = self.J
        dj = self.dj
        
        idx_j1 = []
        idx_j2 = []
        
        # k1 = 1
        # k2 = 1
        # j1 <= j2 <= min(j1+dj,J-1)
        for j1 in range(J):
            for j2 in range(j1,min(j1+dj+1,J)):
                idx_j1.append(j1)
                idx_j2.append(j2)
        
        idx_wph = dict()
        idx_wph['j1'] = torch.tensor(idx_j1).type(torch.long)
        idx_wph['j2'] = torch.tensor(idx_j2).type(torch.long)
        
        return idx_wph

    def _type(self, _type, devid=None):
        if devid is not None:
            if self.chunk_id < self.nb_chunks:
                self.hatpsi = self.hatpsi.to(devid)
            else:
                self.hatpsi = self.hatpsi.to(devid)
                self.hatphi = self.hatphi.to(devid)
                if self.haspsi0:
                    self.hatpsi0 = self.hatpsi0.to(devid)
        else:
            if self.chunk_id < self.nb_chunks:
                self.hatpsi = self.hatpsi.type(_type)
            else:
                self.hatpsi = self.hatpsi.type(_type)
                self.hatphi = self.hatphi.type(_type)
                if self.haspsi0:
                    self.hatpsi0 = self.hatpsi0.type(_type)
        #print('in _type',type(self.hatpsi))
        self.pad.padding_module.type(_type)
        return self

    def cuda(self):
        """
            Moves tensors to the GPU
        """
        devid = self.devid
        print('call cuda with devid=', devid)
        assert(devid>=0)
        if self.chunk_id < self.nb_chunks:
            self.this_wph['j1'] = self.this_wph['j1'].type(torch.cuda.LongTensor).to(devid)
            self.this_wph['j2'] = self.this_wph['j2'].type(torch.cuda.LongTensor).to(devid)
            
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
        pad = self.pad

        # denote
        # nb=batch number
        # nc=number of color channels
        # input: (nb,nc,M,N)
        x_c = pad(input) # add zeros to imag part -> (nb,nc,M,N,2)
        hatx_c = fft2_c2c(x_c) # fft2 -> (nb,nc,M,N,2)
        #print('nbchannels',nb_channels)
        nb = hatx_c.shape[0]
        nc = hatx_c.shape[1]
        hatpsi = self.hatpsi # Pa = max_la-min_la+1, (1,Pa,M,N,2)
        assert(nb==1 and nc==1) # for submeanC
        if self.chunk_id < self.nb_chunks:
            nb_channels = self.this_wph['j1'].shape[0]*L2
            Sout = input.new(nb, nc, nb_channels, 1, 1, 2) # (nb,nc,nb_channels,1,1,2)
            idxb = 0
            idxc = 0
            hatx_bc = hatx_c[idxb,idxc,:,:,:] # (M,N,2)
            hatxpsi_bc = cdgmm(hatpsi, hatx_bc) # (J,L2,M,N,2)
            xpsi_bc = ifft2_c2c(hatxpsi_bc) # (J,L2,M,N,2)
            # permute to put angle 2nd to last
            xpsi_bc_ = xpsi_bc.permute(0,2,3,1,4) # (J,M,N,L2,2)
            xpsi_iso_bc_ = torch.fft(xpsi_bc_, 1, normalized=True)  # (J,M,N,L2,2)
            xpsi_iso_bc = xpsi_iso_bc_.permute(0,3,1,2,4) # (J,L2,M,N,2)
            # select j1, et j2, P_c = number of |j1| in this chunk
            xpsi_bc_j1 = torch.index_select(xpsi_iso_bc, 0, self.this_wph['j1']) # (P_c,L2,M,N,2)
            xpsi_bc_j2 = torch.index_select(xpsi_iso_bc, 0, self.this_wph['j2']) # (P_c,L2,M,N,2)
            # compute empirical cov
            corr_xpsi_bc = mulcu(xpsi_bc_j1,conjugate(xpsi_bc_j2)) # (P_c,L2,M,N,2)
            corr_bc = torch.mean(torch.mean(corr_xpsi_bc,-2,True),-3,True) # (P_c,L2,1,1,2), better numerical presision?!
            Sout[idxb,idxc,:,:,:,:] = corr_bc.view(nb_channels,1,1,2) # (P_c*L2,1,1,2)
        else:
            # ADD 1 chennel for spatial phiJ
            hatxphi_c = cdgmm(hatx_c, self.hatphi) # (nb,nc,M,N,2)
            xphi_c = ifft2_c2c(hatxphi_c)
            # submean from spatial M N
            xphi0_c = self.subinitmeanJ(xphi_c)
            xphi0_mod = self.modulus(xphi0_c) # (nb,nc,M,N,2)
            xphi0_mod2 = mulcu(xphi0_mod,xphi0_mod) # (nb,nc,M,N,2)
            if self.haspsi0:
                nb_channels = 2 + self.dj*L2
                Sout = input.new(nb, nc, nb_channels, 1, 1, 2)
                Sout[:,:,0,:,:,:] = torch.mean(torch.mean(xphi0_mod2,-2,True),-3,True)
                hatxpsi00_c = cdgmm(hatx_c, self.hatpsi0)
                xpsi00_c = ifft2_c2c(hatxpsi00_c)
                xpsi00_mod = self.modulus(xpsi00_c) # (nb,nc,M,N,2)
                xpsi00_mod2 = mulcu(xpsi00_mod,xpsi00_mod) # (nb,nc,M,N,2)
                Sout[:,:,1,:,:,:] = torch.mean(torch.mean(xpsi00_mod2,-2,True),-3,True)
                # add scale interactions with other scales
                idxb = 0
                idxc = 0
                hatx_bc = hatx_c[idxb,idxc,:,:,:] # (M,N,2)
                hatxpsi_bc = cdgmm(hatpsi, hatx_bc) # (J,L2,M,N,2)
                xpsi_bc = ifft2_c2c(hatxpsi_bc) # (J,L2,M,N,2)
                # permute to put angle 2nd to last
                xpsi_bc_ = xpsi_bc.permute(0,2,3,1,4) # (J,M,N,L2,2)
                xpsi_iso_bc_ = torch.fft(xpsi_bc_, 1, normalized=True)  # (J,M,N,L2,2)
                xpsi_iso_bc = xpsi_iso_bc_.permute(0,3,1,2,4) # (J,L2,M,N,2)
                xpsi_bc_la1 = xpsi_iso_bc[0:self.dj,:,:,:,:] # (dj,L2,M,N,2)
                xpsi_bc_la2 = xpsi00_c[idxb,idxc,:,:,:].expand_as(xpsi_bc_la1)
                corr_xpsi_bc = mulcu(xpsi_bc_la1,xpsi_bc_la2) # (dj,L2,M,N,2)
                corr_bc = torch.mean(torch.mean(corr_xpsi_bc,-2,True),-3,True) # (dj,L2,1,1,2)
                Sout[idxb,idxc,2:,:,:,:] = corr_bc.view(self.dj*L2,1,1,2)
            else:
                Sout = input.new(nb, nc, 1, 1, 1, 2)
                Sout[:,:,0,:,:,:] = torch.mean(torch.mean(xphi0_mod2,-2,True),-3,True)
            
        return Sout
        
    def __call__(self, input):
        return self.forward(input)
