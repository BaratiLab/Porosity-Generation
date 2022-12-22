import numpy as np
from PIL import Image
import torch
from torch.autograd import Variable
import torch.nn.functional as F


def prepare_padding_size(M, N, J):
    M_padded = ((M + 2 ** (J)) // 2 ** J + 1) * 2 ** J
    N_padded = ((N + 2 ** (J)) // 2 ** J + 1) * 2 ** J

    return M_padded, N_padded


def cast(Psi, Phi, _type):
    for key, item in enumerate(Psi):
        for key2, item2 in Psi[key].items():
            if torch.is_tensor(item2):
                Psi[key][key2] = Variable(item2.type(_type))
    Phi = [Variable(v.type(_type)) for v in Phi]
    return Psi, Phi


def pad(input, J):
    out_ = F.pad(input, (2 ** J,) * 4, mode='reflect').unsqueeze(input.dim())
    return torch.cat([out_, Variable(input.data.new(out_.size()).zero_())], 4)


def add_imaginary_part(input_tensor):
    temp = input_tensor.unsqueeze(input_tensor.dim())
    return torch.cat([temp, Variable(input_tensor.data.new(temp.size()).zero_())], 4)


def unpad(input, cplx=False):
    if cplx:
        unpadded_input = input[..., 1:-1, 1:-1, :].contiguous()
    else:
        unpadded_input = input[..., 1:-1, 1:-1]
    return unpadded_input


def cdgmm(A, B):
    C = Variable(A.data.new(A.size()))

    A_r = A[..., 0].contiguous().view(-1, A.size(-2) * A.size(-3))
    A_i = A[..., 1].contiguous().view(-1, A.size(-2) * A.size(-3))

    B_r = B[..., 0].contiguous().view(B.size(-2) * B.size(-3)).unsqueeze(0).expand_as(A_i)
    B_i = B[..., 1].contiguous().view(B.size(-2) * B.size(-3)).unsqueeze(0).expand_as(A_r)

    C[..., 0] = (A_r * B_r - A_i * B_i).view(A.shape[:-1])
    C[..., 1] = (A_r * B_i + A_i * B_r).view(A.shape[:-1])
    return C


def periodize(input, k):
    return input.view(input.size(0), input.size(1),
                      k, input.size(2) // k,
                      k, input.size(3) // k,
                      2).mean(4).squeeze(4).mean(2).squeeze(2)


def modulus(input):
    norm = input.norm(p=2, dim=-1, keepdim=True)
    return torch.cat([norm, Variable(norm.data.new(norm.size()).zero_())], -1)


def load(filename):
    return np.ascontiguousarray(Image.open(filename), dtype=np.uint8)


def periodic_dis(i1, i2, per):
    if i2 > i1:
        return min(i2-i1, i1-i2+per)
    else:
        return min(i1-i2, i2-i1+per)


def periodic_signed_dis(i1, i2, per):
    if i2 < i1:
        return i2 - i1 + per
    else:
        return i2 - i1


def rgb2yuv(dataset):
    r = dataset[:, 0, :, :]
    g = dataset[:, 1, :, :]
    b = dataset[:, 2, :, :]
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    u = -0.09991 * r + -0.33609 * g + 0.436 * b
    v = 0.615 * r + -0.55861 * g + -0.05639 * b

    return np.stack([y, u, v], axis=1)


def mean_std(X, cmplx_torch_tensor=False):

    if cmplx_torch_tensor:
        X = X.numpy()
        X = X[..., 0] + 1j * X[..., 1]
    else:
        if type(X) == torch.Tensor:
            X = X.numpy()
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return mean, std


def standardize_feature(X, mean, std, eps=0, cmplx_torch_tensor=False):
    if cmplx_torch_tensor:
        X = X.numpy()
        X = X[..., 0] + 1j * X[..., 1]
    else:
        if type(X) == torch.Tensor:
            X = X.numpy()

    std_feature = (X - mean)/(std + eps)

    if cmplx_torch_tensor:
        std_feature = torch.cat([torch.FloatTensor(np.real(std_feature)).unsqueeze(-1),
                                 torch.FloatTensor(np.imag(std_feature)).unsqueeze(-1)], -1)
    else:
        if type(X) == torch.Tensor:
            std_feature = torch.FloatTensor(std_feature)

    return std_feature


"""
Not used for the moment

def rgb2covariant_color(dataset):
    r = dataset[:, 0, :, :]
    g = dataset[:, 1, :, :]
    b = dataset[:, 2, :, :]
    c_1 = r + g + b
    c_2 = r - b
    c_3 = g - b
    c_4 = r - g
    c_5 = r - 2*g + b
    c_6 = -2*r + g + b
    c_7 = r + g - 2*b

    return np.stack([c_1, c_2, c_3, c_4, c_5, c_6, c_7], axis=1)


def apply_transform(dataset, transform, resize, resize_size): #Assumes a yuv dataset

    if resize:
        dataset_tf = torch.FloatTensor(size=dataset.shape[0:2]+resize_size)
    else:
        dataset_tf = torch.FloatTensor(size=dataset.shape)

    c_0 = dataset[:, 0, :, :]
    c_1 = dataset[:, 1, :, :]
    c_2 = dataset[:, 2, :, :]

    for i in range(dataset.shape[0]):
        c_0_PIL_i = Image.fromarray(c_0[i].astype('uint8'))
        c_1_PIL_i = Image.fromarray(c_1[i].astype('uint8'))
        c_2_PIL_i = Image.fromarray(c_2[i].astype('uint8'))

        c_0_transformed = torch.FloatTensor(np.array(transform(c_0_PIL_i)))
        c_1_transformed = torch.FloatTensor(np.array(transform(c_1_PIL_i)))
        c_2_transformed = torch.FloatTensor(np.array(transform(c_2_PIL_i)))

        dataset_tf[i, 0] = c_0_transformed[0]
        dataset_tf[i, 1] = c_1_transformed[0]
        dataset_tf[i, 2] = c_2_transformed[0]

    return dataset_tf
"""





