# same scale cross-locations correlations
# spatial shift by 0 <= dn1 <= delta_n, -delta_n <= dn2 <= delta_n
# each chunk is one scale, from 0..J-1, last chunk is scale J + psi0
# chunk_id = scale
# j1=j2, k1=1, k2=1

__all__ = ['PhaseHarmonics2d']

import warnings
import math
import torch
import numpy as np
import scipy.io as sio
from .backend import cdgmm, Modulus, fft, \
    Pad, SubInitSpatialMeanCinFFT, PhaseHarmonics2, PeriodicShift2D, mulcu, conjugate
from .filter_bank import filter_bank
from .utils import fft2_c2c, ifft2_c2c, periodic_dis

class WaveletCovIsoFFTShift2d(object):
    # nb_chunks = J, so that each dn can be applied to each chunk with the same shift,
    # chunk_id is the scale parameter j
    def __init__(self, M, N, J, L, delta_n, delta_mode, nb_chunks, chunk_id, devid=0, filname='bumpsteerableg', filid=1):
        self.M, self.N, self.J, self.L = M, N, J, L # size of image, max scale, number of angles [0,pi]
        self.dn = delta_n # shift along dn1 and dn2
        self.dn_mode = delta_mode # 0 for n' = n + dn, 1 for n' = n + 2^j * dn 
        assert(nb_chunks == J)
        self.nb_chunks = nb_chunks # number of chunks to cut cov matrix
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
        check_for_nan = False # True
        self.modulus = Modulus()
        self.pad = Pad(0, pre_pad = self.pre_pad)
        self.phase_harmonics = PhaseHarmonics2.apply
        self.filters_tensor()
        if self.chunk_id < self.nb_chunks:
            self.preselect_filters()
        else:
            self.subinitmeanJ = SubInitSpatialMeanCinFFT()
        self.pershifts = []
        dn = self.dn
        dn_mode = self.dn_mode
        j = self.chunk_id
        if dn_mode==1:
            dn_step = 2**j #2 ^j
        else:
            dn_step = 1
        print('this fftshift2d ', j, ' has step ', dn_step)
        M = self.M
        N = self.N
        dn_loc = []
        for dn1 in range(0,dn+1):
            for dn2 in range(-dn,dn+1):
                if dn1**2+dn2**2 <= self.dn**2 and (dn1!=0 or dn2!=0):
                    # tau=(-dn1,-dn2)
                    Midx = (-dn1)%M
                    Nidx = (-dn2)%N
                    dn_loc.append(Midx*N+Nidx)
        self.dn_loc = torch.tensor(dn_loc).type(torch.long)
        
    def preselect_filters(self):
        # only use thoses filters in the this_wph list
        M = self.M
        N = self.N
        J = self.J
        L = self.L
        L2 = L*2
        hatpsi_la = self.hatpsi.view(J,L2,M,N,2) # (J,L2,M,N,2)
        j = self.chunk_id
        self.hatpsi_pre = hatpsi_la[j,:,:,:,:].view(1,L2,M,N,2) # Pa = L2
        
    def filters_tensor(self):
        J = self.J
        L = self.L
        L2 = L*2
        
        assert(self.M == self.N)
        filpath = './matlab/filters/' + self.filname + str(self.filid) + '_fft2d_N'\
                  + str(self.N) + '_J' + str(self.J) + '_L' + str(self.L) + '.mat'
        matfilters = sio.loadmat(filpath)
        print('filter loaded:', filpath)
        
        if 'filt_fftpsi0' in matfilters:
            fftpsi0 = matfilters['filt_fftpsi0'].astype(np.complex_)
            hatpsi0 = np.stack((np.real(fftpsi0), np.imag(fftpsi0)), axis=-1)
            self.hatpsi0 = torch.FloatTensor(hatpsi0) # (M,N,2)
            self.haspsi0 = True
            print('compute psi0')
        
        fftphi = matfilters['filt_fftphi'].astype(np.complex_)
        hatphi = np.stack((np.real(fftphi), np.imag(fftphi)), axis=-1)
        
        fftpsi = matfilters['filt_fftpsi'].astype(np.complex_)
        hatpsi = np.stack((np.real(fftpsi), np.imag(fftpsi)), axis=-1)
        
        self.hatpsi = torch.FloatTensor(hatpsi) # (J,L2,M,N,2)
        self.hatphi = torch.FloatTensor(hatphi) # (M,N,2)

    def _type(self, _type, devid=None):
        if devid is None:
            if self.chunk_id < self.nb_chunks:
                self.hatpsi_pre = self.hatpsi_pre.type(_type)
            else:
                self.hatphi = self.hatphi.type(_type)
                if self.haspsi0:
                    self.hatpsi0 = self.hatpsi0.type(_type)
        else:
            if self.chunk_id < self.nb_chunks:
                self.hatpsi_pre = self.hatpsi_pre.to(devid)
            else:
                self.hatphi = self.hatphi.to(devid)
                if self.haspsi0:
                    self.hatpsi0 = self.hatpsi0.to(devid)
        self.pad.padding_module.type(_type)
        return self

    def cuda(self):
        """
            Moves tensors to the GPU
        """
        devid = self.devid
        print('call cuda with devid=', devid)
        assert(devid>=0)
        self.dn_loc = self.dn_loc.type(torch.cuda.LongTensor).to(devid)
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
        dn = self.dn
        # denote
        # nb=batch number
        # nc=number of color channels
        # input: (nb,nc,M,N)
        x_c = pad(input) # add zeros to imag part -> (nb,nc,M,N,2)
        hatx_c = fft2_c2c(x_c) # fft2 -> (nb,nc,M,N,2)
        nb = hatx_c.shape[0]
        nc = hatx_c.shape[1]
        assert(nb==1 and nc==1) # for submeanC
        if self.chunk_id < self.nb_chunks:
            nb_channels = L2 * len(self.dn_loc)
        else:
            nb_channels = len(self.dn_loc)
            if self.haspsi0:
                nb_channels += len(self.dn_loc)
        Sout = input.new(nb, nc, nb_channels, 1, 1, 2) # (nb,nc,nb_channels,1,1,2)
        if self.chunk_id < self.nb_chunks:
            nbc = L2
            hatpsi_pre = self.hatpsi_pre
            hatx_bc = hatx_c[0,0,:,:,:] # (M,N,2)
            hatxpsi_bc = cdgmm(hatpsi_pre, hatx_bc) # (1,L2,M,N,2)
            hatxpsi_bc_ = hatxpsi_bc.permute(0,2,3,1,4)
            hatxpsi_iso_bc_ = torch.fft(hatxpsi_bc_, 1, normalized=True)
            hatxpsi_iso_bc = hatxpsi_iso_bc_.permute(0,3,1,2,4)
            hatcorr_bc = mulcu(hatxpsi_iso_bc,conjugate(hatxpsi_iso_bc)) # (1,L2,M,N,2)
            corr_bc = ifft2_c2c(hatcorr_bc)/(M*N) # (1,L2,M,N,2)
            corr_bc_ = corr_bc.view(nbc,M*N,2)
            # keep only tau in dn_loc
            #for pid in range(len(self.dn_loc)):
            #    Sout[0,0,pid*nbc:(pid+1)*nbc,0,0,:] = corr_bc_[0,:,self.dn_loc[pid],:]
            Sout[0,0,:,0,0,:] = torch.index_select(corr_bc_,1,self.dn_loc).view(nbc*len(self.dn_loc),2)
            
        else:
            # ADD 1 chennel for spatial phiJ
            hatxphi_c = cdgmm(hatx_c, self.hatphi) # (nb,nc,M,N,2)
            hatxphi0_c = self.subinitmeanJ(hatxphi_c) # (nb,nc,M,N,2)
            hatphicorr0_c = mulcu(hatxphi0_c,conjugate(hatxphi0_c))
            corrphi0_c = ifft2_c2c(hatphicorr0_c)/(M*N) # (nb,nc,M,N,2)
            corrphi0_c_ = corrphi0_c.view(M*N,2)
            Sout[0,0,0:len(self.dn_loc),0,0,:] = torch.index_select(corrphi0_c_,0,self.dn_loc).view(len(self.dn_loc),2)
            if self.haspsi0:
                hatxpsi00_c = cdgmm(hatx_c, self.hatpsi0)
                hatpsicorr00_c = mulcu(hatxpsi00_c,conjugate(hatxpsi00_c))
                corrpsi00_c = ifft2_c2c(hatpsicorr00_c)/(M*N) # (nb,nc,M,N,2)
                corrpsi00_c_ = corrpsi00_c.view(M*N,2)
                Sout[0,0,len(self.dn_loc):,0,0,:] = torch.index_select(corrpsi00_c_,0,self.dn_loc).view(len(self.dn_loc),2)
                
        return Sout
        
    def __call__(self, input):
        return self.forward(input)

