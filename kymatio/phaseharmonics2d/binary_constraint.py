# x(u), 1<=u<=d,
# ||x||_1 = d, ||x||_2 = d
# gives x(u) = 1 or -1

__all__ = ['PhaseHarmonics2d']

import warnings
import torch

from .backend import SubInitSpatialMeanR

class BinaryConstraint(object):
    def __init__(self):
        return

    def forward(self,input):
        # input: (nb,nc,N,N)
        # output: (nb,nc*2,1,1)
        nb = input.shape[0]
        nc = input.shape[1]
        Sout = input.new(nb,nc*2,1,1)
        absx = torch.abs(input)
        l1x = torch.mean(torch.mean(absx,-1,True),-2,True)
        Sout[:,0:nc,0,0] = l1x
        x2 = input * input
        l2x = torch.mean(torch.mean(x2,-1,True),-2,True)
        Sout[:,nc:,0,0] = l2x
        return Sout

    def __call__(self, input):
        return self.forward(input)


class BinaryConstraint0(object):
    def __init__(self):
        self.subinitmean = SubInitSpatialMeanR()
        return

    def forward(self,input):
        # input: (nb,nc,N,N)
        # output: (nb,nc,1,1)
        nb = input.shape[0]
        nc = input.shape[1]
        Sout = input.new(nb,nc,1,1)
        absx = torch.abs(input)
        absx0 = self.subinitmean(absx)
        x2 = absx0 * absx0
        l2x = torch.mean(torch.mean(x2,-1,True),-2,True)
        Sout[:,:,0,0] = l2x
        return Sout

    def __call__(self, input):
        return self.forward(input)
