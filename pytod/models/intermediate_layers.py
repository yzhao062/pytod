# -*- coding: utf-8 -*-

import time
import numpy as np
import torch
from torch import cdist

from pyod.utils.data import generate_data
from pyod.utils.data import evaluate_print
from pyod.utils.utility import get_list_diff

from itertools import combinations
from ..utils.utility import Standardizer, get_batch_index

from .basic_operators import cdist, cdist_cpu, bottomk, bottomk_cpu
from mpmath import mp, mpf


# machine_eps = mpf(2**-53)

def get_bounded_error(max_value, dimension, machine_eps=np.finfo(float).eps,
                      two_sided=True):
    mp.dps = 100
    factor = (1 + machine_eps) ** (mp.log(dimension) + 2) - 1
    if two_sided:
        return float(2 * (4 * dimension * (max_value ** 2) * factor))
    else:
        return float(4 * dimension * (max_value ** 2) * factor)


def neighbor_within_range(X, range_threshold):
    # calculate the cdist in lower precision
    distance_mat = torch.cdist(X.cuda(), X.cuda()).cpu()
    print(distance_mat)
    # identify the indice pairs
    clear_indices = torch.nonzero((distance_mat <= range_threshold),
                                  as_tuple=False)
    return clear_indices


def neighbor_within_range_low_prec(X, range_threshold):
    n_samples, n_features = X.shape[0], X.shape[1]

    # get the error bound
    error_bound = float(
        get_bounded_error(torch.max(X).cpu().numpy(), n_features))

    # calculate the cdist in lower precision
    distance_mat = torch.cdist(X.cuda().half(), X.cuda().half()).cpu()

    # selected_indice = torch.nonzero(distance_mat<= threshold, as_tuple=False)

    # we can calculate diffence instead
    # see code here
    full_indices = np.arange(0, n_samples)

    # identify the ambiguous indice pairs
    amb_indices = torch.nonzero(
        (distance_mat <= range_threshold + error_bound) &
        (distance_mat >= range_threshold - error_bound), as_tuple=False)

    print(amb_indices.shape)
    print(torch.unique(amb_indices[:, 0]).shape)

    # these are the indices of the samples with no ambiguity
    clear_indices = get_list_diff(full_indices,
                                  amb_indices[:, 0].cpu().numpy())

    # find the matched samples 
    clear_pairs = torch.nonzero(
        distance_mat[clear_indices, :] <= range_threshold, as_tuple=False)
    print('initial pairs', clear_pairs.shape)

    # recalculate for the amb_indices
    # get the samples for recalculation
    amb_1 = X[amb_indices[:, 0], :]
    amb_2 = X[amb_indices[:, 1], :]

    pdist = torch.nn.PairwiseDistance(p=2)
    amb_dist = pdist(amb_1, amb_2)

    # finally return a 2-d tensor containing all the tensors
    true_neigh_indices = torch.nonzero((amb_dist <= range_threshold),
                                       as_tuple=True)
    print(true_neigh_indices[0].shape)

    clear_pairs = torch.cat(
        (clear_pairs, amb_indices[true_neigh_indices[0], :]))
    print('ultimate true pairs', clear_pairs.shape)

    print('imprecision pairs correct:',
          amb_indices.shape[0] - true_neigh_indices[0].shape[0])
    return clear_pairs


def neighbor_within_range_low_prec_float(X, range_threshold):
    n_samples, n_features = X.shape[0], X.shape[1]

    # get the error bound
    error_bound = float(
        get_bounded_error(torch.max(X).cpu().numpy(), n_features))

    # calculate the cdist in lower precision
    distance_mat = torch.cdist(X.cuda(), X.cuda()).cpu()

    # selected_indice = torch.nonzero(distance_mat<= threshold, as_tuple=False)

    # we can calculate diffence instead
    # see code here
    full_indices = np.arange(0, n_samples)

    # identify the ambiguous indice pairs
    amb_indices = torch.nonzero(
        (distance_mat <= range_threshold + error_bound) &
        (distance_mat >= range_threshold - error_bound), as_tuple=False)

    print(amb_indices.shape)
    print(torch.unique(amb_indices[:, 0]).shape)

    # these are the indices of the samples with no ambiguity
    clear_indices = get_list_diff(full_indices,
                                  amb_indices[:, 0].cpu().numpy())

    # find the matched samples 
    clear_pairs = torch.nonzero(
        distance_mat[clear_indices, :] <= range_threshold, as_tuple=False)
    print('initial pairs', clear_pairs.shape)

    # recalculate for the amb_indices
    # get the samples for recalculation
    amb_1 = X[amb_indices[:, 0], :]
    amb_2 = X[amb_indices[:, 1], :]

    pdist = torch.nn.PairwiseDistance(p=2)
    amb_dist = pdist(amb_1, amb_2)

    # finally return a 2-d tensor containing all the tensors
    true_neigh_indices = torch.nonzero((amb_dist <= range_threshold),
                                       as_tuple=True)
    print(true_neigh_indices[0].shape)

    clear_pairs = torch.cat(
        (clear_pairs, amb_indices[true_neigh_indices[0], :]))
    print('ultimate true pairs', clear_pairs.shape)

    print('imprecision pairs correct:',
          amb_indices.shape[0] - true_neigh_indices[0].shape[0])
    return clear_pairs


def get_indices_clear_pairs(clear_pairs, sample_indice):
    # find pairs neighbors for specific samples
    return clear_pairs[
        torch.nonzero((clear_pairs[:, 0] == sample_indice), as_tuple=False), 1]


def knn_full(A, B, k=5, p=2.0, device=None):
    dist_c = cdist(A, B, p=p)
    btk_d, btk_i = bottomk(dist_c, k=k)
    return btk_d, btk_i

def knn_full_cpu(A, B, k=5, p=2.0):
    dist_c = cdist_cpu(A, B, p=p)
    btk_d, btk_i = bottomk_cpu(dist_c, k=k)
    return btk_d, btk_i


def knn_batch_intermediate(A, B, k=5, p=2.0, batch_size=None):
    # this is the map step
    n_samples, n_features = A.shape[0], A.shape[1]
    n_distance = B.shape[0]

    if batch_size >= n_samples:
        return knn_full(A, B, k, p)

    batch_index_A = get_batch_index(n_samples, batch_size)
    batch_index_B = get_batch_index(n_distance, batch_size)
    print(batch_index_A)
    print(batch_index_B)

    n_batch_A = len(batch_index_A)
    n_batch_B = len(batch_index_B)

    # this is a cpu tensor to save space
    # cdist_mat = torch.zeros([n_samples, n_distance])
    k_dist_mat = torch.zeros([n_samples, n_batch_B * k])
    k_inds_mat = torch.zeros([n_samples, n_batch_B * k]).int()

    for i, index_A in enumerate(batch_index_A):
        for j, index_B in enumerate(batch_index_B):
            # get the dist
            cdist_mat_batch = torch.cdist(A[index_A[0]:index_A[1], :].cuda(),
                                          B[index_B[0]:index_B[1], :].cuda(),
                                          p=p)

            # important, need to select from the batch index
            # otherwise the ind starts from 0 again
            batch_inds = torch.arange(index_B[0], index_B[1]).repeat(
                batch_size, 1)
            # print(batch_inds.shape)

            bk = bottomk(cdist_mat_batch, k)
            # we need a global indices here
            k_dist_mat[i * batch_size:(i + 1) * batch_size,
            j * k:(j + 1) * k] = bk[0]
            k_inds_mat[i * batch_size:(i + 1) * batch_size,
            j * k:(j + 1) * k] = batch_inds.gather(1, bk[1].long())

    return k_dist_mat, k_inds_mat


def get_knn_from_intermediate(intermediate_knn, k):
    # this is the reduce step

    # sort distance for index, real knn happens here
    sorted_ind = torch.argsort(intermediate_knn[0], dim=1)

    # bottomk_indices.gather(1, bottomk_indices_argsort)

    # selected the first k for each sample
    # gather适合每个sample选择的row都不一样。
    knn_dist = intermediate_knn[0].gather(1, sorted_ind[:, :k])
    knn_inds = intermediate_knn[1].gather(1, sorted_ind[:, :k])

    return knn_dist, knn_inds


def knn_batch(A, B, k=5, p=2.0, batch_size=None):
    intermediate_knn = knn_batch_intermediate(A, B, k, p, batch_size)
    return get_knn_from_intermediate(intermediate_knn, k)


def get_cosine_similarity(input1, input2, use_cuda=False):
    # todo: fix use cuda 
    # torch.sum(nn_1* nn_2, dim=1) / (torch.linalg.norm(nn_1, dim=1)**2 * torch.linalg.norm(nn_2, dim=1)**2)
    return torch.sum(input1 * input2, dim=1) / (
                torch.linalg.norm(input1, dim=1) ** 2 *
                torch.linalg.norm(input2, dim=1) ** 2)

# n_train = 20000  # number of training points
# n_features = 100
# batch_size = 10000
# p = 2
# k = 5

# # # Generate sample data
# # # X_train = torch.randn([n_train, n_features]).half()
# # # X_train = torch.randn([n_train, n_features])
# A = torch.randn([n_train, n_features])
# # # X_train_norm = Standardizer(X_train, return_mean_std=False)
# B = A


# # intermediate_knn = knn_batch_intermediate(A, B, k, batch_size=batch_size)
# # knn_dist, knn_inds = get_knn_from_intermediate(intermediate_knn, k)

# knn_dist, knn_inds = knn_batch(A, B, k, batch_size=batch_size)

# %%

# btk_d, btk_i = knn_full(A, B, k)

# non_equal = torch.nonzero(btk_d!=knn_dist, as_tuple=False)
# non_equal_ind = torch.nonzero(btk_i!=knn_inds, as_tuple=False)

# # make sure the indices are the same for
# assert (non_equal_ind.shape[0] == 0)
# # # it is wiered though.
# # for i in range(non_equal.shape[0]):
# #     print(non_equal[i], btk_d[non_equal[i][0], non_equal[i][1]], knn_dist[non_equal[i][0], non_equal[i][1]])
