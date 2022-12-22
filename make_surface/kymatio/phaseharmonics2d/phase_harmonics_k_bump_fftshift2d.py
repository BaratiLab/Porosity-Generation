
__all__ = ['PhaseHarmonics2d']

import warnings
import math
import torch
import numpy as np
import scipy.io as sio
#import torch.nn.functional as F
from .backend import cdgmm, Modulus, fft, \
    Pad, SubInitSpatialMeanC, PhaseHarmonicsIso, \
    mulcu, conjugate, DivInitStd
from .filter_bank import filter_bank
from .utils import fft2_c2c, ifft2_c2c, periodic_dis

class PhaseHarmonics2d(object):
    def __init__(self, M, N, J, L, delta_n, maxk_shift, nb_chunks, chunk_id,\
                 devid=0, submean=1, stdnorm=0, outmode=0):
        # only for k=0 on the wavelets (j<=J)
        # and for k=1 for the basse freq (j=J+1) 
        # with dx!=0, dy!=0, in [-Delta,Delta]^2
        # nb_chunks = J+1
        self.M, self.N, self.J, self.L = M, N, J, L
        # size of image, max scale, number of angles [0,pi]
        self.dn = delta_n
        assert(nb_chunks == J+1)
        self.K = maxk_shift + 1
        self.k = torch.arange(0, self.K).type(torch.float) # vector between [0,..,K-1]
        self.nb_chunks = nb_chunks # number of chunks to cut whp cov
        self.chunk_id = chunk_id
        self.devid = devid
        self.submean = submean
        self.stdnorm = stdnorm
        assert( self.chunk_id <= self.nb_chunks )
        self.pre_pad = False # no padding
        self.cache = False # cache filter bank
        self.nbcov = 0 # count complex number twice
        self.build()

    def build(self):
        check_for_nan = False # True
        
        dn = self.dn
        sj = min(self.chunk_id,self.J-1) # j start from zero
        dn_step = 2**sj # subsampling rate
        print('chunk_id = ', self.chunk_id, ' has downsampling rate= ', dn_step)
        M = self.M
        N = self.N
        dn_loc = []
        for dn1 in range(0,dn+1):
            for dn2 in range(-dn,dn+1):
                if (dn1!=0 or dn2!=0):
                    # tau=(-dn1,-dn2)
                    Midx = (-dn1*dn_step)%M
                    Nidx = (-dn2*dn_step)%N
                    dn_loc.append(Midx*N+Nidx)
        self.dn_loc = torch.tensor(dn_loc).type(torch.long)

        self.modulus = Modulus()
        self.pad = Pad(0, pre_pad = self.pre_pad, pad_mode='Reflect') # default is zero padding
        self.phase_harmonics = PhaseHarmonicsIso.apply
        self.M_padded, self.N_padded = self.M, self.N
        self.filters_tensor()
        self.idx_wph = self.compute_idx()
        self.this_wph = self.get_this_chunk(self.nb_chunks, self.chunk_id)
        if self.submean == 1:
            self.subinitmean = SubInitSpatialMeanC()
            self.subinitmeanJ = SubInitSpatialMeanC()
            if self.stdnorm == 1:
                self.divinitstd = DivInitStd()
                self.divinitstdJ = DivInitStd()

    def filters_tensor(self):
        # TODO load bump steerable wavelets
        J = self.J
        L = self.L
        L2 = L*2

        assert(self.M == self.N)
        matfilters = sio.loadmat('./make_surface/matlab/filters/bumpsteerableg1_fft2d_N' +\
                                 str(self.N) + '_J' + str(self.J) + '_L' + str(self.L) + '.mat')

        fftphi = matfilters['filt_fftphi'].astype(np.complex_)
        hatphi = np.stack((np.real(fftphi), np.imag(fftphi)), axis=-1)

        fftpsi = matfilters['filt_fftpsi'].astype(np.complex_)
        hatpsi = np.stack((np.real(fftpsi), np.imag(fftpsi)), axis=-1)

        self.hatpsi = torch.FloatTensor(hatpsi) # (J,L2,M,N,2)
        self.hatphi = torch.FloatTensor(hatphi) # (M,N,2)

    def get_this_chunk(self, nb_chunks, chunk_id):
        this_wph = dict()
        if chunk_id < nb_chunks-1:
            nb_cov_chunk = np.zeros(nb_chunks,dtype=np.int32)
            for idxc in range(nb_chunks-1):
                nb_cov_chunk[idxc] = self.L*2*self.K
            assert(self.L*2*self.K == len(self.idx_wph['la1'])/self.J)
            nb_cov = len(self.idx_wph['la1'])
            offset = int(0)
            for idxc in range(nb_chunks-1):
                if idxc == chunk_id:
                    this_wph['la1'] = self.idx_wph['la1'][offset:offset+nb_cov_chunk[idxc]]
                    this_wph['la2'] = self.idx_wph['la2'][offset:offset+nb_cov_chunk[idxc]]
                offset = offset + nb_cov_chunk[idxc]
            print('this chunk', chunk_id, ' size is ', len(this_wph['la1']), ' among ', nb_cov)
        return this_wph
    
    def compute_idx(self):
        L = self.L
        L2 = L*2
        J = self.J
        K = self.K
        idx_la1 = []
        idx_la2 = []
        
        # j1=j2, ell1=ell2, k1=k2<=maxk_shift
        # only count (dn1,dn2) != (0,0)
        for j1 in range(J):
            for ell1 in range(L2):
                j2 = j1
                ell2 = ell1
                for k1 in range(K):
                    k2 = k1
                    idx_la1.append(K*L2*j1 + K*ell1 + k1)
                    idx_la2.append(K*L2*j2 + K*ell2 + k2)
                    if k1==0:
                        self.nbcov += len(self.dn_loc)
                    else:
                        self.nbcov += 2*len(self.dn_loc)

        # low pass
        self.nbcov += len(self.dn_loc)
        
        idx_wph = dict()
        idx_wph['la1'] = torch.tensor(idx_la1).type(torch.long)
        idx_wph['la2'] = torch.tensor(idx_la2).type(torch.long)
        
        return idx_wph

    def _type(self, _type, devid=None):
        self.hatpsi = self.hatpsi.type(_type)
        self.hatphi = self.hatphi.type(_type)
        self.k = self.k.type(_type)
        if devid is not None:
            self.hatpsi = self.hatpsi.to(devid)
            self.hatphi = self.hatphi.to(devid)
            self.k = self.k.to(devid)
        self.pad.padding_module.type(_type)
        
        return self

    def cuda(self):
        """
            Moves tensors to the GPU
        """
        devid = self.devid
        print('call cuda with devid=', devid)
        assert(devid>=0)
        if self.chunk_id < self.nb_chunks - 1:
            self.this_wph['la1'] = self.this_wph['la1'].type(torch.cuda.LongTensor).to(devid)
            self.this_wph['la2'] = self.this_wph['la2'].type(torch.cuda.LongTensor).to(devid)
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
        dn = self.dn
        K = self.K
        k = self.k # # vector between [0,..,K-1]
        pad = self.pad

        x_c = pad(input) # add zeros to imag part -> (nb,nc,M,N,2)
        hatx_c = fft2_c2c(x_c) # fft2 -> (nb,nc,M,N,2)     
        nb = hatx_c.shape[0]
        nc = hatx_c.shape[1]
        hatpsi_la = self.hatpsi # (J,L2,M,N,2)
        assert(nb==1 and nc==1) # otherwise fix submeanC
        if self.chunk_id < self.nb_chunks - 1:
            nbc = self.this_wph['la1'].shape[0]
            nb_channels = nbc * len(self.dn_loc)
        else:
            nb_channels = len(self.dn_loc)
        Sout = input.new(nb, nc, nb_channels, 1, 1, 2)

        idxb = 0 # since nb=1
        idxc = 0 # since nc=1, otherwise use loop
        hatx_bc = hatx_c[idxb,idxc,:,:,:] # (M,N,2)
        if self.chunk_id < self.nb_chunks - 1:
            hatxpsi_bc = cdgmm(hatpsi_la, hatx_bc) # (J,L2,M,N,2)
            # ifft , then compute phase harmonics along k
            xpsi_bc = ifft2_c2c(hatxpsi_bc)
            xpsi_bc = xpsi_bc.unsqueeze(-2) # (J,L2,M,N,1,2)
            xpsi_ph_bc = self.phase_harmonics(xpsi_bc, k) # (J,L2,M,N,K,2)
            # permute K to 3rd dimension (for indexing)
            xpsi_wph_bc = xpsi_ph_bc.permute(0,1,4,2,3,5).contiguous() # (J,L2,K,M,N,2)
            # sub spatial mean for all channels
            if self.submean==1:
                xpsi_wph_bc0 = self.subinitmean(xpsi_wph_bc)
                if self.stdnorm==1:
                    xpsi_wph_bc0 = self.divinitstd(xpsi_wph_bc0)
            else:
                xpsi_wph_bc0 = xpsi_wph_bc

            # reshape to (1,J*L2*K,M,N,2)
            xpsi_wph_bc0_ = xpsi_wph_bc0.view(1,J*L2*K,M,N,2)
            xpsi_bc_la1 = torch.index_select(xpsi_wph_bc0_, 1, self.this_wph['la1']) # (1,P_c,M,N,2)
            # select la1, et la2, P_c = number of |la1| in this chunk
            # WE ASSUME THAT THEY ARE THE SAME
            hatxpsi_bc_la1 = fft2_c2c(xpsi_bc_la1) # fft2 -> (1,P_c,M,N,2)
            hatxpsi_bc_la2 = hatxpsi_bc_la1 # torch.index_select(hatxpsi_bc, 1, self.this_wph['la2']) # (1,P_c,M,N,2)
            hatcorr_bc = mulcu(hatxpsi_bc_la1,conjugate(hatxpsi_bc_la2))
            corr_bc = ifft2_c2c(hatcorr_bc)/(M*N) # (1,P_c,M,N,2)
            corr_bc_ = corr_bc.view(nbc,M*N,2)
            # keep only tau in dn_loc
            Sout[0,0,:,0,0,:] = torch.index_select(corr_bc_,1,self.dn_loc).view(nbc*len(self.dn_loc),2)
        else:
            hatxphi_c = cdgmm(hatx_c, self.hatphi) # (nb,nc,M,N,2)
            xphi_c = ifft2_c2c(hatxphi_c)
            if self.submean==1:
                xphi0_c = self.subinitmeanJ(xphi_c)
                if self.stdnorm==1:
                    xphi0_c = self.divinitstdJ(xphi0_c)
            else:
                xphi0_c = xphi_c
            hatxphi0_c = fft2_c2c(xphi0_c)
            hatphicorr0_c = mulcu(hatxphi0_c,conjugate(hatxphi0_c))
            corrphi0_c = ifft2_c2c(hatphicorr0_c)/(M*N) # (nb,nc,M,N,2)
            corrphi0_c_ = corrphi0_c.view(M*N,2)
            Sout[0,0,0:len(self.dn_loc),0,0,:] = torch.index_select(corrphi0_c_,0,self.dn_loc).view(len(self.dn_loc),2)
            
        return Sout
        
    def __call__(self, input):
        return self.forward(input)
