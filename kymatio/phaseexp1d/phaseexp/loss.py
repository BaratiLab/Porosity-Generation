import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from scattering.scattering1d.utils import modulus
except ImportError:
    from scatwave.scattering1d.utils import modulus

def mse_norm(input, detail=False):
    sq_err = modulus(input) ** 2
    if detail:
        return sq_err
    return torch.mean(sq_err.view(-1))


def mse_loss(input, target, detail=False):
    return mse_norm(input - target, detail=detail)
    

def l1_loss(input, target, detail=False):
    abs_err = modulus(input - target)
    if detail:
        return abs_err
    return torch.mean(abs_err.view(-1))


class LogMSELoss(object):
    def __init__(self, epsilon=1e-3):
        super(LogMSELoss, self).__init__()
        self.eps = epsilon

    def __call__(self, input, target):
        sq_err = modulus(input - target) ** 2
        loss = torch.log(sq_err / self.eps + 1)
        
        return torch.mean(loss.view(-1))


class RelativeMSELoss(object):
    def __init__(self, epsilon=1e-3):
        super(RelativeMSELoss, self).__init__()
        self.eps = epsilon

    def __call__(self, input, target):
        sq_err = modulus(input - target) ** 2
        target_norm = modulus(target) ** 2
        loss = sq_err / (target_norm + self.eps)

        return torch.mean(loss.view(-1))


def PSNR(input, target, A):
    """input and target are expected to be numpy arrays with same shape.
    A: values in target should be contained in [-A/2, A/2]"""

    l2_err = np.mean(np.abs(input - target) ** 2)
    psnr = 10 * np.log10(A ** 2 / l2_err)
    return psnr
