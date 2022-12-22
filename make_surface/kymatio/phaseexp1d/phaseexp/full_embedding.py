import gc
import sys
if __name__ == '__main__':
    sys.path.append('../pyscatwave/')
import math
from itertools import product
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, grad
import complex_utils as cplx
from metric import PhaseHarmonicPrunedSelect, PhaseHarmonicPrunedSeparated
from scattering.scattering1d import Scattering1D
from global_const import Tensor

class FullEmbedding(nn.Module):
    '''
    Class that combines the Scattering and Phase Harmonic embeddings, allowing
    to select which types of coefficients to use.

    Inputs:
    '''

    def __init__(self, T, J, Q, phe_params={}, scatt_params={},
                 scatt_orders=(0,1,2), phe_coeffs=['harmonic', 'mixed']):
        '''
        Parameters:
        T: temporal support of inputs
        J: number of octaves
        Q: number of voices per octave
        phe_params: dictionary of kwargs for PhaseHarmonic embedding
        scatt_params: dictionary of kwargs for scattering embedding
        '''
        super(FullEmbedding, self).__init__()
        self.T = T
        self.Q = Q
        self.scatt_orders = scatt_orders
        self.phe_coeffs = phe_coeffs

        self.is_cuda_p = False
        self.compute_scatt_snd_p = 2 in scatt_orders

        # Remove from user input  parameters that are imposed below
        scatt_params = {key:value for key, value in scatt_params.items()
                        if key not in ['order2', 'average_U1', 'average_U2',
                                       'precision']}
        if 'J' in scatt_params:
            J_scat = scatt_params['J']
            del scatt_params['J']
        else:
            J_scat = J
        if 'J' in phe_params:
            J_phe = phe_params['J']
            del phe_params['J']
        else:
            J_phe = J
        self.J_scat = J_scat
        self.J_phe = J_phe

        self.scattering = Scattering1D(T, J_scat, Q, order2=self.compute_scatt_snd_p,
                                       average_U1=True, average_U2=False,
                                       vectorize=False, precision='double',
                                       **scatt_params)

        # self.phase_harmonic = PhaseHarmonicPrunedSelect(
        #     J, Q, T, coeff_select=self.phe_coeffs, **phe_params)
        self.phase_harmonic = PhaseHarmonicPrunedSeparated(
            J_phe, Q, T, coeff_select=self.phe_coeffs, **phe_params)

    def cuda(self):
        super(FullEmbedding, self).cuda()
        self.scattering = self.scattering.cuda()
        self.phase_harmonic = self.phase_harmonic.cuda()
        self.is_cuda_p = True
        return self

    def cpu(self):
        super(FullEmbedding, self).cpu()
        self.scattering = self.scattering.cpu()
        self.phase_harmonic = self.phase_harmonic.cpu()
        self.is_cuda_p = False
        return self

    def count_scatt_coeffs(self):
        meta = Scattering1D.compute_meta_scattering(self.J_scat, self.Q, order2=True)
        num_coeff = 0
        for ord in self.scatt_orders:
            num_coeff += len([key for key,value in meta.items()
                              if value['order'] == str(ord)])
        return num_coeff

    def shape(self):
        '''
        Return number of coefficients on both embeddings
        '''
        return self.phase_harmonic.num_coeff() + self.count_scatt_coeffs()

    def split_scatt_coefs(self, S):
        # Select coefficients: scattering
        output = [[], [], []]
        self.scatt_keys = (
            [0],
            [key for key in S.keys() if type(key) is tuple and len(key)==2],
            [key for key in S.keys() if type(key) is tuple and len(key)==4])
        for ord in self.scatt_orders:
            for key in self.scatt_keys[ord]:
                output[ord].append(S[key])
        return output

    def format(self, x):
        # TODO: assert size [1,1,N]
        if len(x.shape) == 3:  # pytorch  shape, [1, 1, N]
            x_scat = x.double() #.float()
            # Add complex dimension for phe
            x_phe = torch.cat((x.unsqueeze(-1),
                               torch.zeros_like(x).unsqueeze(-1)),
                              dim=-1).double()
        else:  # PHE shape, [1,1,N,2]
            # Remove complex dimension for scatt:
            x_scat = x[...,0].double() #.float()
            x_phe = x.double()
        # Scattering gives error if requires_grad is false
        #x_scat = torch.Tensor(x_scat, requires_grad=False)
        #x_phe  = Variable(x_phe, requires_grad=False)
        return x_scat, x_phe

    def forward(self, x):
        # Adapt data formats
        x_scat, x_phe = self.format(x)

        # Compute both embeddings
        emb_sca = self.scattering.forward(x_scat)
        emb_phe = self.phase_harmonic.forward(x_phe)

        scat0, scat1, scat2 = self.split_scatt_coefs(emb_sca)

        if len(scat1) > 0:
            scat1_l1 = torch.cat([torch.norm(coef, p=1, dim=-1)
                                 for coef in scat1], dim=0)
            scat1_l2 = torch.cat([torch.norm(coef, p=2, dim=-1) ** 2
                                 for coef in scat1], dim=0)
            scat1 = torch.stack((scat1_l1, scat1_l2), dim=-1)
            # scat1 is of size [1,1,ncoef,1,2], with last dimension containing l1 and l2 norms

        else:
            scat1 = Tensor([])
        if len(scat2) > 0:
            scat2 = torch.cat([torch.norm(coef, p=2, dim=-1) ** 2
                                 for coef in scat2],dim=0)
        else:
            scat2 = Tensor([])

        if self.is_cuda_p:
            scat1 = scat1.cuda()
            scat2 = scat2.cuda()

        if not scat1.shape == torch.Size([0]):
            scat1 = scat1.unsqueeze(0).unsqueeze(0)
            scat1 = torch.cat((scat1, torch.zeros_like(scat1)), dim=-2)
        if not scat2.shape == torch.Size([0]):
            scat2 = scat2.unsqueeze(0).unsqueeze(0)
            scat2 = torch.cat((scat2, torch.zeros_like(scat2)), dim=-1)

        # print(scat1.shape)
        # print(scat2.shape)
        # print(emb_phe[0].shape)
        # print(emb_phe[1].shape)

        # fst_ord = torch.cat((scat1, emb_phe[0]), dim=2)
        # snd_ord = torch.cat((scat2, emb_phe[1]), dim=2)

        fst_ord = [scat1, emb_phe[0]]
        snd_ord = [scat2, emb_phe[1]]
        #gc.collect()
        return fst_ord, snd_ord

if __name__ == '__main__':
    use_cuda = True

    T = 2**10
    J = 8
    Q = 1
    x = torch.randn((1, 1, T))

    emb = FullEmbedding(T, J, Q)
    if use_cuda:
        emb = emb.cuda()
        x = x.cuda()

    fst, snd = emb(x)
