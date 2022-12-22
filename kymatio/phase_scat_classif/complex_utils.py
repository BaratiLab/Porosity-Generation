from utils import modulus
import torch
import numpy as np


def real(z):
    return z[..., 0]


def imag(z):
    return z[..., 1]


def conjugate(z):
    z_copy = z.clone()
    z_copy[..., 1] = -z_copy[..., 1]
    return z_copy


def mul(z1, z2):
    zr = real(z1) * real(z2) - imag(z1) * imag(z2)
    zi = real(z1) * imag(z2) + imag(z1) * real(z2)
    z = torch.stack((zr, zi), dim=-1)
    return z


def ones_like(z):
    re = torch.ones_like(z[..., 0])
    im = torch.zeros_like(z[..., 1])
    return torch.stack((re, im), dim=-1)


def phase(z):
    x, y = real(z), imag(z)
    z_mod = modulus(z)[..., 0]

    eps = 1e-32
    mask_real_neg = (torch.abs(y) <= eps) * (x <= 0)
    mask_zero = z_mod <= eps

    theta = torch.atan2(y, x)

    theta.masked_fill_(mask_real_neg, np.pi)
    theta.masked_fill_(mask_zero, 0.)

    return theta


def complex_log(z, eps):
    phase_z = phase(z).unsqueeze(-1)
    z_mod = modulus(z)[..., 0].unsqueeze(-1)
    log_z_mod = torch.log(z_mod + eps)
    complex_log = torch.cat([log_z_mod, phase_z], -1)

    return complex_log


def phase_exp(z, k):
    z_mod = modulus(z)[...,0].unsqueeze(-1)
    e_iphi = z / z_mod
    e_iphi.masked_fill_(z_mod == 0, 0)
    if k == 0:
        e_ikphi = ones_like(e_iphi)
    else:
        e_ikphi = e_iphi
        for i in range(1, k):
            e_ikphi = mul(e_ikphi, e_iphi)


    return z_mod*e_ikphi