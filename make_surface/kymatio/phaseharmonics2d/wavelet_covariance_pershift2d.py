# same scale pershift correlations
# shifted by dn1^2 + dn2^2 <= delta_n, dn1 and dn2 !=0
# each chunk is one scale, from scale 0..J-1 and J
# chunk_id = scale

__all__ = ['PhaseHarmonics2d']

import warnings
import math
import torch
import numpy as np
import scipy.io as sio
from .backend import cdgmm, Modulus, fft, \
    Pad, SubInitSpatialMeanC, PhaseHarmonics2, PeriodicShift2D, mulcu, conjugate
from .filter_bank import filter_bank
from .utils import fft2_c2c, ifft2_c2c, periodic_dis

class WaveletCovPerShift2d(object):
    # nb_chunks = J, so that each dn can be applied to each chunk with the same shift,
    # chunk_id is the scale parameter j
    def __init__(self, M, N, J, L, delta_n, delta_mode, nb_chunks, chunk_id, devid=0, filname='bumpsteerableg', filid=1):
        self.M, self.N, self.J, self.L = M, N, J, L # size of image, max scale, number of angles [0,pi]
        self.dn = delta_n # shift along dn1 and dn2
        self.dl = 0 # delta_l # max angular interactions
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
        if self.dl > self.L:
            raise (ValueError('delta_l must be <= L'))

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
            self.idx_wph = self.compute_idx()
            self.this_wph = self.get_this_chunk(self.nb_chunks, self.chunk_id)
            self.this_wph_size = torch.numel(self.this_wph['la1'])
            self.preselect_filters()
        else:
            self.subinitmeanJ = SubInitSpatialMeanC()
        self.pershifts = []
        dn = self.dn
        dn_mode = self.dn_mode
        j = self.chunk_id
        if dn_mode==1:
            dn_step = 2**j #2 ^j
        else:
            dn_step = 1
        print('this pershift2d ', j, ' has step ', dn_step)
        for dn1 in range(0,dn+1):
            for dn2 in range(-dn,dn+1):
                if dn1**2+dn2**2 <= self.dn**2 and (dn1!=0 or dn2!=0):
                    self.pershifts.append(PeriodicShift2D(self.M,self.N,dn_step*dn1,dn_step*dn2))
        
    def preselect_filters(self):
        # only use thoses filters in the this_wph list
        M = self.M
        N = self.N
        J = self.J
        L = self.L
        L2 = L*2
        min_la1 = self.this_wph['la1'].min()
        max_la1 = self.this_wph['la1'].max()
        min_la2 = self.this_wph['la2'].min()
        max_la2 = self.this_wph['la2'].max()
        min_la = min(min_la1,min_la2)
        max_la = max(max_la1,max_la2)
        print('this la range',min_la,max_la)
        hatpsi_la = self.hatpsi.view(1,J*L2,M,N,2) # (J,L2,M,N,2) -> (1,J*L2,M,N,2)
        self.hatpsi_pre = hatpsi_la[:,min_la:max_la+1,:,:,:] # Pa = max_la-min_la+1, (1,Pa,M,N,2)
        self.this_wph['la1_pre'] = self.this_wph['la1'] - min_la
        self.this_wph['la2_pre'] = self.this_wph['la2'] - min_la
    
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

        #print('filter shapes')
        #print(self.hatpsi.shape)
        #print(self.hatphi.shape)

    def get_this_chunk(self, nb_chunks, chunk_id):
        # cut self.idx_wph into smaller pieces
        nb_cov = len(self.idx_wph['la1'])
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
                this_wph['la1'] = self.idx_wph['la1'][offset:offset+nb_cov_chunk[idxc]]
                this_wph['la2'] = self.idx_wph['la2'][offset:offset+nb_cov_chunk[idxc]]
                this_wph['k1'] = self.idx_wph['k1'][:,offset:offset+nb_cov_chunk[idxc],:,:]
                this_wph['k2'] = self.idx_wph['k2'][:,offset:offset+nb_cov_chunk[idxc],:,:]
            offset = offset + nb_cov_chunk[idxc]

        print('this chunk', chunk_id, ' size is ', len(this_wph['la1']), ' among ', nb_cov)
        return this_wph

    def compute_ncoeff(self):
        # return number of mean (nb1) and cov (nb2) of all idx
        L = self.L
        L2 = L*2
        J = self.J
        dl = self.dl
        
        hit_nb1 = dict() # hat Ix counts, true zero is not counted
        hit_nb2 = dict() # hat Cx counts, complex value is counted twice
        
        # j1=j2, k1=1, k2=1
        for j1 in range(J):
            j2 = j1
            for ell1 in range(L2):
                for ell2 in range(L2):
                    if periodic_dis(ell1, ell2, L2) <= dl:
                        k1 = 1
                        k2 = 1
                        hit_nb1[(j1,k1,ell1)]=0
                        hit_nb1[(j2,k2,ell2)]=0
                        hit_nb2[(j1,k1,ell1,j2,k2,ell2)] = 2
                        
        #print('hit nb1 values',list(hit_nb1.values()))
        nb1 = np.array(list(hit_nb1.values()), dtype=int).sum()
        nb2 = np.array(list(hit_nb2.values()), dtype=int).sum() 

        return nb1, nb2
    
    def compute_idx(self):
        L = self.L
        L2 = L*2
        J = self.J
        dl = self.dl

        idx_la1 = []
        idx_la2 = []
        idx_k1 = []
        idx_k2 = []

        # j1=j2, k1=1, k2=1
        for j1 in range(J):
            j2 = j1
            for ell1 in range(L2):
                for ell2 in range(L2):
                    if periodic_dis(ell1, ell2, L2) <= dl:
                        k1 = 1
                        k2 = 1
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
        if self.chunk_id < self.nb_chunks:
            self.this_wph['k1'] = self.this_wph['k1'].type(torch.cuda.FloatTensor).to(devid)
            self.this_wph['k2'] = self.this_wph['k2'].type(torch.cuda.FloatTensor).to(devid)
            self.this_wph['la1_pre'] = self.this_wph['la1_pre'].type(torch.cuda.LongTensor).to(devid)
            self.this_wph['la2_pre'] = self.this_wph['la2_pre'].type(torch.cuda.LongTensor).to(devid)
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
        dl = self.dl
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
            nb_channels = self.this_wph['la1_pre'].shape[0] * len(self.pershifts)
        else:
            nb_channels = len(self.pershifts)
            if self.haspsi0:
                nb_channels += len(self.pershifts)
        Sout = input.new(nb, nc, nb_channels, 1, 1, 2) # (nb,nc,nb_channels,1,1,2)
        if self.chunk_id < self.nb_chunks:
            nbc = self.this_wph['la1_pre'].shape[0]
            hatpsi_pre = self.hatpsi_pre
            for idxb in range(nb):
                for idxc in range(nc):
                    hatx_bc = hatx_c[idxb,idxc,:,:,:] # (M,N,2)
                    hatxpsi_bc = cdgmm(hatpsi_pre, hatx_bc) # (1,Pa,M,N,2)
                    xpsi_bc = ifft2_c2c(hatxpsi_bc) # (1,Pa,M,N,2)
                    xpsi_bc_la1 = torch.index_select(xpsi_bc, 1, self.this_wph['la1_pre']) # (1,P_c,M,N,2)
                    for pid in range(len(self.pershifts)):
                        pershift = self.pershifts[pid]
                        y_c = pershift(x_c)
                        haty_c = fft2_c2c(y_c)
                        haty_bc = haty_c[idxb,idxc,:,:,:]
                        hatypsi_bc = cdgmm(hatpsi_pre, haty_bc) # (1,Pa,M,N,2)
                        ypsi_bc = ifft2_c2c(hatypsi_bc) # (1,Pa,M,N,2)
                        # select la1, et la2, P_c = number of |la1| in this chunk = self.this_wph_size
                        ypsi_bc_la2 = torch.index_select(ypsi_bc, 1, self.this_wph['la2_pre']) # (1,P_c,M,N,2)
                        # compute empirical cov
                        corr_xpsi_bc = mulcu(xpsi_bc_la1,conjugate(ypsi_bc_la2)) # (1,P_c,M,N,2)
                        corr_bc = torch.mean(torch.mean(corr_xpsi_bc,-2,True),-3,True) # (1,P_c,1,1,2)
                        Sout[idxb,idxc,pid*nbc:(pid+1)*nbc,:,:,:] = corr_bc[0,:,:,:,:]
        else:
            # ADD 1 chennel for spatial phiJ
            hatxphi_c = cdgmm(hatx_c, self.hatphi) # (nb,nc,M,N,2)
            xphi_c = ifft2_c2c(hatxphi_c)
            xphi0_c = self.subinitmeanJ(xphi_c) # submean from spatial M N
            for pid in range(len(self.pershifts)):
                pershift = self.pershifts[pid]
                yphi0_c = pershift(xphi0_c) # self.subinitmeanJ(yphi_c) # SAME mean as xphi_c
                xyphi0_c = mulcu(xphi0_c,yphi0_c) # (nb,nc,M,N,2)
                Sout[:,:,pid,:,:,:] = torch.mean(torch.mean(xyphi0_c,-2,True),-3,True)

            if self.haspsi0:
                hatxpsi00_c = cdgmm(hatx_c, self.hatpsi0)
                xpsi00_c = ifft2_c2c(hatxpsi00_c)
                for pid in range(len(self.pershifts)):
                    pershift = self.pershifts[pid]
                    ypsi00_c = pershift(xpsi00_c)
                    xypsi00_c = mulcu(xpsi00_c,ypsi00_c) # (nb,nc,M,N,2)
                    Sout[:,:,pid+len(self.pershifts),:,:,:] = torch.mean(torch.mean(xypsi00_c,-2,True),-3,True)
                
        return Sout
        
    def __call__(self, input):
        return self.forward(input)
