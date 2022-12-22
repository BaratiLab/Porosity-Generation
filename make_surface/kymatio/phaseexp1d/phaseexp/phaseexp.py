import sys
import global_const
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, Function
if __name__ == "__main__":
    import os
    sys.path.append(os.path.join(global_const.CODEPATH, '../pyscatwave'))
try:
    from scattering.scattering1d.utils import modulus
except ImportError:
    from scatwave.scattering1d.utils import modulus
import complex_utils as cplx
from utils import is_long_tensor, is_double_tensor
from utils import HookDetectNan, masked_fill_zero


class PhaseExp(nn.Module):
    def __init__(self, K, k_type='linear', keep_k_dim=False, check_for_nan=False):
        super(PhaseExp, self).__init__()
        self.K = K
        self.keep_k_dim = keep_k_dim
        self.check_for_nan = check_for_nan
        assert k_type in ['linear', 'log2']
        self.k_type = k_type

    def forward(self, z):
        s = z.size()
        z_mod = modulus(z)  # modulus

        eitheta = cplx.phaseexp(z, z_mod.unsqueeze(-1))  # phase

        # compute phase exponent : |z| * exp(i k theta)
        if self.k_type == 'linear':
            eiktheta = cplx.pows(eitheta, self.K - 1, dim=1)
        elif self.k_type == 'log2':
            eiktheta = cplx.log2_pows(eitheta, self.K - 1, dim=1)
        z_pe = z_mod.unsqueeze(-1) * eiktheta

        if not self.keep_k_dim:
            z_pe = z_pe.view(s[0], -1, *s[2:])

        if z.requires_grad and self.check_for_nan:
            z.register_hook(HookDetectNan("z in PhaseExp"))
            if self.K > 1:
                z_mod.register_hook(HookDetectNan("z_mod in PhaseExp"))
            eitheta.register_hook(HookDetectNan("eitheta in PhaseExp"))
            eiktheta.register_hook(HookDetectNan("eiktheta in PhaseExp"))
            z_pe.register_hook(HookDetectNan("z_pe in PhaseExp"))

        return z_pe


class PhaseHarmonic(nn.Module):
    def __init__(self, check_for_nan=False):
        super(PhaseHarmonic, self).__init__()
        self.check_for_nan = check_for_nan

    def forward(self, z, k):
        # check type ok k, move to float
        if not is_long_tensor(k):
            raise TypeError("Expected torch(.cuda).LongTensor but got {}".format(k.type()))
        if is_double_tensor(z):
            k = k.double()
        else:
            k = k.float()

        s = z.size()
        z_mod = modulus(z)  # modulus

        # compute phase
        theta = cplx.phase(z)  # phase
        k = k.unsqueeze(0).unsqueeze(1)
        for spatial_dim in theta.size()[2:-1]:
            k = k.unsqueeze(-1)
        ktheta = k * theta
        eiktheta = torch.stack((torch.cos(ktheta), torch.sin(ktheta)), dim=-1)

        # compute phase exponent : |z| * exp(i k theta)
        z_pe = z_mod.unsqueeze(-1) * eiktheta

        if z.requires_grad and self.check_for_nan:
            z.register_hook(HookDetectNan("z in PhaseExp"))
                # , torch.stack((z_mod, z_mod), dim=-1), torch.stack((theta, theta), dim=-1)))
            z_mod.register_hook(HookDetectNan("z_mod in PhaseExp"))
            eiktheta.register_hook(HookDetectNan("eiktheta in PhaseExp"))
            z_pe.register_hook(HookDetectNan("z_pe in PhaseExp"))

        return z_pe


class StablePhaseHarmonic(Function):
    @staticmethod
    def forward(ctx, z, k, eps=1e-64):
        z = z.detach()
        if not is_long_tensor(k):
            raise TypeError("Expected torch(.cuda).LongTensor but got {}".format(k.type()))
        if is_double_tensor(z):
            k = k.double()
        else:
            k = k.float()

        # modulus, real and imaginary parts
        r = z.norm(p=2, dim=-1)
        x, y = cplx.real(z), cplx.imag(z)

        # mask where NaNs appear
        mask_zero = r <= eps
        mask_real_neg = (torch.abs(y) == 0) * (x <= 0)

        # phase
        theta = torch.atan2(y, x)
        # theta = torch.atan(y / (r + x)) * 2

        # phase exponent
        k = k.unsqueeze(0).unsqueeze(1)
        for spatial_dim in theta.size()[2:-1]:
            k = k.unsqueeze(-1)
        ktheta = k * theta
        eiktheta = torch.stack((torch.cos(ktheta), torch.sin(ktheta)), dim=-1)
        z_pe = r.unsqueeze(-1) * eiktheta

        k_dupl = k.repeat([1] * (r.dim() - 1) + [r.size(-1)])
        z_pe[..., 0][mask_real_neg] = r[mask_real_neg] * torch.cos(k_dupl[mask_real_neg] * np.pi)
        z_pe[..., 1].masked_fill_(mask_real_neg, 0)
        z_pe[mask_zero] = z[mask_zero]
        ctx.save_for_backward(x, y, r, k, z_pe, mask_zero, mask_real_neg)

        # print(
        #     mask_zero.cpu().numpy()[0, 0, 0], mask_real_neg.cpu().numpy()[0, 0, 0],
        #     z.cpu().numpy()[0, 0, 0], theta.cpu().numpy()[0, 0, 0], z_pe.cpu().numpy()[0, 0, 0]
        # )

        return z_pe

    @staticmethod
    def backward(ctx, grad_output):
        x, y, r, k, z_pe, mask_zero, mask_real_neg = ctx.saved_tensors

        # some intermediate variables
        x_tilde = r + x
        e = x_tilde ** 2 + y ** 2

        # derivative with respect to the real part
        dtdxr = x / (r ** 2) 
        dtdxi = (-k * y * x_tilde / (r * e)) * 2

        # derivative with respect to the imaginary part
        dtdyr = y / (r ** 2)
        dtdyi = (k * x * x_tilde / (r * e)) * 2

        # stack to get gradient
        dtdx = torch.stack((dtdxr, dtdxi), dim=-1)
        dtdy = torch.stack((dtdyr, dtdyi), dim=-1)

        dtdx = cplx.mul(dtdx, z_pe)
        dtdy = cplx.mul(dtdy, z_pe)

        # handles NaNs on R-
        k_dupl = k.repeat([1] * (r.dim() - 1) + [r.size(-1)])
        k_nan = k_dupl[mask_real_neg]
        dtdx[..., 0][mask_real_neg] = -torch.cos(k_nan * np.pi)
        dtdx[..., 1].masked_fill_(mask_real_neg, 0)
        dtdy[..., 0].masked_fill_(mask_real_neg, 0)
        dtdy[..., 1][mask_real_neg] = -k_nan * torch.cos(k_nan * np.pi)

        # handle NaNs when modulus is 0
        dtdx[..., 0].masked_fill_(mask_zero, 0)
        dtdx[..., 1].masked_fill_(mask_zero, 0)
        dtdy[..., 0].masked_fill_(mask_zero, 0)
        dtdy[..., 1].masked_fill_(mask_zero, 0)

        dtdx = (dtdx * grad_output).sum(-1)
        dtdy = (dtdy * grad_output).sum(-1)

        grad_input = torch.stack((dtdx, dtdy), dim=-1)

        return grad_input, None, None


if __name__ == "__main__":
    from tqdm import tqdm
    from torch.autograd import gradcheck

    nexp = 1
    batch = 1
    size = 1
    cuda = False

    ph = PhaseHarmonic(check_for_nan=True)
    # ph = StablePhaseHarmonic.apply

    seeds = [np.random.randint(2 ** 20) for _ in range(nexp)]
    for seed in tqdm(seeds):
        tqdm.write('Random seed used : {}'.format(seed))
        # set random seed
        np.random.seed(seed)
        torch.manual_seed(seed + 1)
        torch.cuda.manual_seed(seed + 2)

        zt = torch.randn(batch, 1, size, 2)
        kt = torch.randint(16, (size,))
        print(kt)
        if cuda:
            zt = zt.cuda()
            kt = kt.cuda()

        zt[..., 0] = -4
        zt[..., 1] = 0

        z = Variable(zt.double(), requires_grad=True)
        k = Variable(kt.long())

        def test(z):
            return ph(z, k)
        inpt = (z,)

        tqdm.write(str(gradcheck(test, inpt, eps=1e-8, atol=1e-4)))
