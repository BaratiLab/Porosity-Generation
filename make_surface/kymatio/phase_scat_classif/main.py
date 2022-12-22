import torch
import torchvision
from utils import rgb2yuv
from representation_complex import compute_scat, compute_phase_harmonic_cor, compute_phase_harmonic_compl, \
    compute_modulus_cor, compute_mixed_coeffs, compute_phase_harmonic_2nd_order, compute_phase_harmonic_compl_2nd_order, \
    compute_phase_harmonic_cor_color, compute_phase_harmonic_color_compl

from complex_utils import complex_log
from utils import mean_std, standardize_feature
from sklearn.linear_model import LogisticRegression

import numpy as np


# Define dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
testset= torchvision.datasets.CIFAR10(root='./data', train=False, download=True)

train_data = getattr(trainset, 'train_data').transpose(0, 3, 1, 2)
test_data = getattr(testset, 'test_data').transpose(0, 3, 1, 2)

train_data = rgb2yuv(train_data)
test_data = rgb2yuv(test_data)

train_data = torch.FloatTensor(train_data)
test_data = torch.FloatTensor(test_data)

labels_train = np.array(getattr(trainset, 'train_labels'))
labels_test = np.array(getattr(testset, 'test_labels'))

#classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Subsample dataset
train_sub_factor = 1
test_sub_factor = 1
nb_samples_train = labels_train.shape[0]
nb_samples_test = labels_test.shape[0]

X_train = train_data[0:int(train_sub_factor * nb_samples_train)]
y_train = labels_train[0:int(train_sub_factor * nb_samples_train)]

X_test = test_data[0:int(test_sub_factor * nb_samples_test)]
y_test = labels_test[0:int(test_sub_factor * nb_samples_test)]


f = open("new_rep_test", "w")

J = 3
L = 8
batch_size = 10000
log = True
normalize = True
C = 0.001
epsilon = 1e-1
delta = 1
l_max = 1


scat_train = compute_scat(X_train, J, L, L, batch_size)
scat_test = compute_scat(X_test, J, L, L, batch_size)


phase_harmonics_train = compute_phase_harmonic_cor(X_train, J, L, delta, l_max, batch_size)
phase_harmonics_test = compute_phase_harmonic_cor(X_test, J, L, delta, l_max, batch_size)

phase_harmonics_compl_train = compute_phase_harmonic_compl(X_train, J, L, delta, l_max, batch_size)
phase_harmonics_compl_test = compute_phase_harmonic_compl(X_test, J, L, delta, l_max, batch_size)

#modulus_train = compute_modulus_cor(X_train, J, L, delta, l_max, batch_size)
#modulus_test = compute_modulus_cor(X_test, J, L, delta, l_max, batch_size)

mixed_coeffs_train = compute_mixed_coeffs(X_train, J, L, delta, l_max, batch_size)
mixed_coeffs_test = compute_mixed_coeffs(X_test, J, L, delta, l_max, batch_size)

"""
phase_2nd_order_train = compute_phase_harmonic_2nd_order(X_train, J, L, L, delta, l_max, batch_size)
phase_2nd_order_test = compute_phase_harmonic_2nd_order(X_test, J, L, L, delta, l_max, batch_size)

phase_2nd_order_compl_train = compute_phase_harmonic_compl_2nd_order(X_train, J, L, L, delta, l_max, batch_size)
phase_2nd_order_compl_test = compute_phase_harmonic_compl_2nd_order(X_test, J, L, L, delta, l_max, batch_size)

phase_color_train = compute_phase_harmonic_cor_color(X_train, J, L, delta, l_max, batch_size)
phase_color_test = compute_phase_harmonic_cor_color(X_test, J, L, delta, l_max, batch_size)

phase_color_compl_train = compute_phase_harmonic_color_compl(X_train, J, L, delta, l_max, batch_size)
phase_color_compl_test = compute_phase_harmonic_color_compl(X_test, J, L, delta, l_max, batch_size)
"""

#X_rep_train_copy = scat_train
#X_rep_test_copy = scat_test


X_rep_train_copy = torch.cat([scat_train, phase_harmonics_train, phase_harmonics_compl_train, mixed_coeffs_train], dim=1)
X_rep_test_copy = torch.cat([scat_test, phase_harmonics_test, phase_harmonics_compl_test, mixed_coeffs_test], dim=1)
"""
print("scat_train.shape:{}".format(scat_train.shape))
print("phase_harmonics_train.shape:{}".format(phase_harmonics_train.shape))
print("phase_harmonics_compl_train.shape:{}".format(phase_harmonics_compl_train.shape))
print("modulus_train.shape:{}".format(modulus_train.shape))
print("mixed_coeffs_train.shape:{}".format(mixed_coeffs_train.shape))
print("phase_2nd_order_train.shape:{}".format(phase_2nd_order_train.shape))
print("phase_2nd_order_compl_train.shape:{}".format(phase_2nd_order_compl_train.shape))
print("phase_color_train.shape:{}".format(phase_color_train.shape))
print("phase_color_compl_train.shape:{}".format(phase_color_compl_train.shape))

"""
# Log normalisation
if log:
    X_rep_train = complex_log(X_rep_train_copy, 1e-6)
    X_rep_test = complex_log(X_rep_test_copy, 1e-6)

else:
    X_rep_train = X_rep_train_copy
    X_rep_test = X_rep_test_copy

# Normalization
if normalize:
    mean, std = mean_std(X_rep_train, True)
    X_rep_train = standardize_feature(X_rep_train, mean, std, cmplx_torch_tensor=True)
    X_rep_test = standardize_feature(X_rep_test, mean, std, cmplx_torch_tensor=True)

    X_rep_train = X_rep_train.reshape(X_rep_train.shape[0], -1).numpy()
    X_rep_test = X_rep_test.reshape(X_rep_test.shape[0], -1).numpy()

    size = X_rep_train.shape[1]

    clf = LogisticRegression(verbose=100, solver='sag', n_jobs=-1, max_iter=100, C=C)
    clf.fit(X_rep_train, y_train)
    score_train = clf.score(X_rep_train, y_train)
    score_test = clf.score(X_rep_test, y_test)

    size = X_rep_train.shape[1]


    print("size: {}".format(size))
    print("Train_accuracy: {}".format(score_train))
    print("Test_accuracy: {}".format(score_test))
    print("Generalization gap: {}".format(score_train - score_test))

    """
    print("size: {}".format(size), file=f)
    print("Train_accuracy: {}".format(score_train), file=f)
    print("Test_accuracy: {}".format(score_test), file=f)
    print("Generalization gap: {}".format(score_train - score_test), file=f)
    """





