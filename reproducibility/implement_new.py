# -*- coding: utf-8 -*-

import warnings

warnings.filterwarnings("ignore")

import os
import sys
import time

import torch
from pytorch_memlab import LineProfiler

# temporary solution for relative imports in case pyod is not installed
# if pyod is installed, no need to use the following line
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname("__file__"), '..')))

from pytod.utils.data import Standardizer
from pytod.utils.utility import validate_device
from pytod.models.intermediate_layers import neighbor_within_range_low_prec, neighbor_within_range,neighbor_within_range_low_prec_float
from pytod.models.basic_operators_batch import cdist_batch

n_train = 50000  # number of training points
n_features = 200
batch_size = 40000

# Generate sample data
X_train = torch.randn([n_train, n_features]).double()
X_train_norm = Standardizer(X_train, return_mean_std=False)
device = validate_device(0)

# start = time.time()
# with LineProfiler(neighbor_within_range) as prof:
#
#     clear_pairs = neighbor_within_range(X_train_norm, range_threshold=12,
#                                         device=device)
#     # clear_pairs = neighbor_within_range(X_train_norm, range_threshold=12,
#     #                                     batch_size=batch_size, device=device)
# print(prof.display())
# end = time.time()
# print('64-bit time', end - start)


# with LineProfiler(cdist_batch) as prof:
#     start = time.time()
#     distance_mat = cdist_batch(X_train_norm, X_train_norm,
#                                batch_size=batch_size, device=device)
#
#     # identify the indice pairs
#     clear_indices = torch.nonzero((distance_mat <= 12),
#                                   as_tuple=False)
#     end = time.time()
#     print('64-bit time', end - start)
# print(prof.display())

# start = time.time()
# with LineProfiler(neighbor_within_range_low_prec_float) as prof:
#
#     clear_pairs = neighbor_within_range_low_prec_float(X_train_norm,
#                                                        range_threshold=12,
#                                                        device=device)
#
#     # clear_pairs = neighbor_within_range_low_prec_float(X_train_norm,
#     #                                                    range_threshold=12,
#     #                                                    batch_size=batch_size,
#     #                                                    device=device)
#
# print(prof.display())
# end = time.time()
# print('32-bit time', end - start)


start = time.time()
with LineProfiler(neighbor_within_range_low_prec) as prof:

    clear_pairs = neighbor_within_range_low_prec(X_train_norm,
                                                 range_threshold=12,
                                                 device=device)
    # clear_pairs = neighbor_within_range_low_prec(X_train_norm,
    #                                              range_threshold=12,
    #                                              batch_size=batch_size,
    #                                              device=device)

print(prof.display())
end = time.time()
print('16-bit time', end - start)