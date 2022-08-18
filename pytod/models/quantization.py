# -*- coding: utf-8 -*-
"""Benchmark of all implemented algorithms
"""

from __future__ import division
from __future__ import print_function

import os
import sys

import torch

# temporary solution for relative imports in case pyod is not installed
# if pyod is installed, no need to use the following line
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname("__file__"), '..')))
# supress warnings for clean output
import warnings

warnings.filterwarnings("ignore")

import numpy as np
from scipy.io import loadmat

# from pyod.models.feature_bagging import FeatureBagging

from pyod.utils.data import evaluate_print

from pyod.utils.utility import standardizer

from basic_operators import topk, bottomk, bottomk_low_prec, topk_low_prec

from mpmath import mp, mpf

machine_eps = mpf(2 ** -53)


def get_bounded_error(max_value, dimension, machine_eps=np.finfo(float).eps,
                      two_sided=True):
    mp.dps = 100
    factor = (1 + machine_eps) ** (mp.log(dimension) + 2) - 1
    if two_sided:
        return float(2 * (4 * dimension * (max_value ** 2) * factor))
    else:
        return float(4 * dimension * (max_value ** 2) * factor)


# print(get_bounded_error(1, 1000000))
# error_bound = float(get_bounded_error(1, 1000000))

# TODO: add neural networks, LOCI, SOS, COF, SOD

# Define data file and read X and y
mat_file_list = [
    # 'annthyroid.mat',
    'arrhythmia.mat',
    # 'breastw.mat',
    # 'glass.mat',
    # 'ionosphere.mat',
    # 'letter.mat',
    # 'lympho.mat',
    # 'mammography.mat',
    # 'mnist.mat',
    # 'musk.mat',

    # 'optdigits.mat',
    # 'pendigits.mat',
    # 'pima.mat',
    # 'satellite.mat',
    # 'satimage-2.mat',
    # # 'shuttle.mat',
    # # 'smtp_n.mat',
    # 'speech.mat',
    # 'thyroid.mat',
    # 'vertebral.mat',
    # 'vowels.mat',
    # 'wbc.mat',
    # 'wine.mat',
]

mat_file = 'speech.mat'
mat = loadmat(os.path.join("datasets", "ODDS", mat_file))

X = mat['X']
y = mat['y'].ravel()

n_samples, n_features = X.shape[0], X.shape[1]

outliers_fraction = np.count_nonzero(y) / len(y)
outliers_percentage = round(outliers_fraction * 100, ndigits=4)

# scaler =  MinMaxScaler(feature_range=((1,2)))

# X_transform = scaler.fit_transform(X)
# a = rankdata(X, axis=0)
# b = rankdata(X_transform, axis=0)

X = standardizer(X)
error_bound = get_bounded_error(np.max(X), n_features)
print(error_bound)

k = 10
# X_train = torch.tensor(X).half().cuda()
X_train = torch.tensor(X).float()
# X_train = torch.tensor(X).double().cuda()


cdist_dist = torch.cdist(X_train, X_train, p=2)

bottomk_dist, bottomk_indices = bottomk(cdist_dist, k)
bottomk_dist1, bottomk_indices1 = bottomk_low_prec(cdist_dist, k)

# bottomk_dist_sorted, bottomk_indices_argsort = torch.sort(bottomk_dist1, dim=1)
# bottomk_indices_sorted = bottomk_indices1.gather(1, bottomk_indices_argsort)
print()
print('bottomk is not sorted...')
# we can only ensure the top k
print(torch.sum((bottomk_dist[:, k - 1] != bottomk_dist1[:, k - 1]).int()))
print(
    torch.sum((bottomk_indices[:, k - 1] != bottomk_indices1[:, k - 1]).int()))

# we can only ensure the top k
print(torch.sum((bottomk_dist != bottomk_dist1).int()))
print(torch.sum((bottomk_indices != bottomk_indices1).int()))

bottomk_dist2, bottomk_indices2 = bottomk_low_prec(cdist_dist, k,
                                                   sort_value=True)
print()
print('bottomk is sorted...')
# we ensure topk
print(torch.sum((bottomk_dist[:, k - 1] != bottomk_dist2[:, k - 1]).int()))
print(
    torch.sum((bottomk_indices[:, k - 1] != bottomk_indices2[:, k - 1]).int()))

# we can ensure all
print(torch.sum((bottomk_dist != bottomk_dist2).int()))
print(torch.sum((bottomk_indices != bottomk_indices2).int()))

# %%

print()
print('topk is not sorted...')

topk_dist, topk_indices = topk(cdist_dist, k)
topk_dist1, topk_indices1 = topk_low_prec(cdist_dist, k)

# we can only ensure the top k
print(torch.sum((topk_dist[:, k - 1] != topk_dist1[:, k - 1]).int()))
print(torch.sum((topk_indices[:, k - 1] != topk_indices1[:, k - 1]).int()))

print(torch.sum((topk_dist != topk_dist1).int()))
print(torch.sum((topk_indices != topk_indices1).int()))

topk_dist2, topk_indices2 = topk_low_prec(cdist_dist, k, sort_value=True)
print()
print('topk is sorted...')
print(torch.sum((topk_dist[:, k - 1] != topk_dist2[:, k - 1]).int()))
print(torch.sum((topk_indices[:, k - 1] != topk_indices2[:, k - 1]).int()))

print(torch.sum((topk_dist != topk_dist2).int()))
print(torch.sum((topk_indices != topk_indices2).int()))

# here we flip the order
decision_scores = bottomk_dist[:, -1]

evaluate_print('knn', y, decision_scores.cpu())

# #%%
# from basic_operators import topk, intersec1d
# from pytorch_memlab import LineProfiler
# from pytorch_memlab import MemReporter
# import time


# # t1 = torch.randint(low=0, high=20000000, size=[20000000])
# # t2 = torch.randint(low=5000000, high=25000000, size=[20000000])

# t1 = torch.rand(size=[50000000])
# t2 = torch.rand(size=[50000000])


# t1, t2 = t1.half().cuda(), t2.half().cuda()
# # t1, t2 = t1.float().cuda(), t2.float().cuda()
# # t1, t2 = t1.double().cuda(), t2.double().cuda()

# def w(A, B):
#     return intersec1d(A, B)

# with LineProfiler(w) as prof:
#     # distance_mat = batch_cdist(X_train_norm, X_train_norm, batch_size=5000)
#     start = time.time()
#     a = w(t1, t2) 
#     end = time.time()
#     print(end - start)

# print(prof.display())


# #%%
# from basic_operators import topk
# from pytorch_memlab import LineProfiler
# from pytorch_memlab import MemReporter
# import time

# def Standardizer(X_train, mean=None, std=None, return_mean_std=False):

#     if mean is None:
#         mean = torch.mean(X_train, axis=0)
#         std = torch.std(X_train, axis=0)
#         # print(mean.shape, std.shape)
#         assert (mean.shape[0] == X_train.shape[1])
#         assert (std.shape[0] == X_train.shape[1])


#     X_train_norm = (X_train-mean)/std
#     assert(X_train_norm.shape == X_train.shape)

#     if return_mean_std:
#         return X_train_norm, mean, std
#     else:
#         return  X_train_norm

# contamination = 0.1  # percentage of outliers
# n_train = 200000  # number of training points
# n_test = 1000  # number of testing points
# n_features = 2000

# # Generate sample data
# X_train, y_train, X_test, y_test = \
#     generate_data(n_train=n_train,
#                   n_test=n_test,
#                   n_features=n_features,
#                   contamination=contamination,
#                   random_state=42)

# k = 5


# X_train = torch.tensor(X_train)
# X_test = torch.tensor(X_test)

# # X_train_norm, X_train_mean, X_train_std = Standardizer(X_train, return_mean_std=True)
# # X_test_norm = Standardizer(X_test, mean=X_train_mean, std=X_train_std)


# # X_train_norm = X_train.half().cuda()
# # X_train_norm = X_train.float().cuda()
# X_train_norm = X_train.double().cuda()
# print(X_train_norm.type())


# def w(A, k):
#     return torch.topk(A, k)

# with LineProfiler(w) as prof:
#     # distance_mat = batch_cdist(X_train_norm, X_train_norm, batch_size=5000)
#     start = time.time()
#     a,b = w(X_train_norm, k) 
#     end = time.time()
#     print(end - start)

# print(prof.display())

# %%
