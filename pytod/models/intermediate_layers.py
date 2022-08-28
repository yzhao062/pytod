# -*- coding: utf-8 -*-

import numpy as np
import torch
from mpmath import mp
from pyod.utils.utility import get_list_diff

from .basic_operators import bottomk
from .basic_operators_batch import cdist_batch
from .functional_operators import knn_full
from ..utils.utility import get_batch_index


# machine_eps = mpf(2**-53)

def get_bounded_error(max_value, dimension, machine_eps=np.finfo(float).eps,
                      two_sided=True):
    factor = (1 + machine_eps) ** (mp.log(dimension) + 2) - 1
    if two_sided:
        return float(2 * (4 * dimension * (max_value ** 2) * factor))
    else:
        return float(4 * dimension * (max_value ** 2) * factor)


def neighbor_within_range_low_prec_float(X, range_threshold, batch_size=None,
                                         device='cpu'):
    n_samples, n_features = X.shape[0], X.shape[1]

    # get the error bound
    error_bound = float(
        get_bounded_error(torch.max(X).cpu().numpy(), n_features))

    # calculate the cdist in lower precision
    # distance_mat = torch.cdist(X.to(device).float(),
    #                            X.to(device).float()).cpu()
    if batch_size is None:
        distance_mat = torch.cdist(X.to(device).float(),
                                   X.to(device).float()).cpu()
    else:
        distance_mat = cdist_batch(X.float(), X.float(), batch_size=batch_size,
                                   device=device)

    # we can calculate difference instead
    # see code here
    full_indices = np.arange(0, n_samples)

    # identify the ambiguous indice pairs
    amb_indices = torch.nonzero(
        (distance_mat <= range_threshold + error_bound) &
        (distance_mat >= range_threshold - error_bound), as_tuple=False)

    # print(amb_indices.shape)
    # print(torch.unique(amb_indices[:, 0]).shape)

    # these are the indices of the samples with no ambiguity
    clear_indices = get_list_diff(full_indices,
                                  amb_indices[:, 0].cpu().numpy())

    # find the matched samples 
    clear_pairs = torch.nonzero(
        distance_mat[clear_indices, :] <= range_threshold, as_tuple=False)
    # print('initial pairs', clear_pairs.shape)

    # recalculate for the amb_indices
    # get the samples for recalculation
    amb_1 = X[amb_indices[:, 0], :]
    amb_2 = X[amb_indices[:, 1], :]

    pdist = torch.nn.PairwiseDistance(p=2)
    amb_dist = pdist(amb_1, amb_2)

    # finally return a 2-d tensor containing all the tensors
    true_neigh_indices = torch.nonzero((amb_dist <= range_threshold),
                                       as_tuple=True)
    # print(true_neigh_indices[0].shape)

    clear_pairs = torch.cat(
        (clear_pairs, amb_indices[true_neigh_indices[0], :]))
    # print('ultimate true pairs', clear_pairs.shape)

    # print('imprecision pairs correct:',
    #       amb_indices.shape[0] - true_neigh_indices[0].shape[0])
    return clear_pairs


def get_indices_clear_pairs(clear_pairs, sample_indice):
    # find pairs neighbors for specific samples
    return clear_pairs[
        torch.nonzero((clear_pairs[:, 0] == sample_indice), as_tuple=False), 1]


def neighbor_within_range(X, range_threshold, batch_size=None, device='cpu'):
    # calculate the cdist in original precision
    if batch_size is None:
        distance_mat = torch.cdist(X.to(device), X.to(device)).cpu()
    else:
        distance_mat = cdist_batch(X, X, batch_size=batch_size, device=device)

    # print(distance_mat)
    # identify the indice pairs
    clear_indices = torch.nonzero((distance_mat <= range_threshold),
                                  as_tuple=False)
    return clear_indices


def neighbor_within_range_low_prec(X, range_threshold, batch_size=None,
                                   device='cpu'):
    n_samples, n_features = X.shape[0], X.shape[1]

    # get the error bound
    error_bound = float(
        get_bounded_error(torch.max(X).cpu().numpy(), n_features))

    # calculate the cdist in lower precision
    # distance_mat = torch.cdist(X.to(device).half(), X.to(device).half()).cpu()

    if batch_size is None:
        distance_mat = torch.cdist(X.to(device).half(),
                                   X.to(device).half()).cpu()
    else:
        distance_mat = cdist_batch(X.half(), X.half(), batch_size=batch_size,
                                   device=device).cpu()

    # we can calculate difference instead
    # see code here
    full_indices = np.arange(0, n_samples)

    # identify the ambiguous indice pairs
    amb_indices = torch.nonzero(
        (distance_mat <= range_threshold + error_bound) &
        (distance_mat >= range_threshold - error_bound), as_tuple=False)

    # print(amb_indices.shape)
    # print(torch.unique(amb_indices[:, 0]).shape)

    # these are the indices of the samples with no ambiguity
    clear_indices = get_list_diff(full_indices,
                                  amb_indices[:, 0].cpu().numpy())

    # find the matched samples
    clear_pairs = torch.nonzero(
        distance_mat[clear_indices, :] <= range_threshold, as_tuple=False)
    # print('initial pairs', clear_pairs.shape)

    # recalculate for the amb_indices
    # get the samples for recalculation
    amb_1 = X[amb_indices[:, 0], :]
    amb_2 = X[amb_indices[:, 1], :]

    pdist = torch.nn.PairwiseDistance(p=2)
    amb_dist = pdist(amb_1, amb_2)

    # finally return a 2-d tensor containing all the tensors
    true_neigh_indices = torch.nonzero((amb_dist <= range_threshold),
                                       as_tuple=True)
    # print(true_neigh_indices[0].shape)

    clear_pairs = torch.cat(
        (clear_pairs, amb_indices[true_neigh_indices[0], :]))
    # print('ultimate true pairs', clear_pairs.shape)

    # print('imprecision pairs correct:',
    #       amb_indices.shape[0] - true_neigh_indices[0].shape[0])
    return clear_pairs


def knn_batch_intermediate(A, B, k=5, p=2.0, batch_size=None, device='cpu'):
    # this is the map step
    n_samples, n_features = A.shape[0], A.shape[1]
    n_distance = B.shape[0]

    if batch_size >= n_samples or batch_size is None:
        return knn_full(A, B, k, p)

    batch_index_A = get_batch_index(n_samples, batch_size)
    batch_index_B = get_batch_index(n_distance, batch_size)
    # print(batch_index_A)
    # print(batch_index_B)

    n_batch_A = len(batch_index_A)
    n_batch_B = len(batch_index_B)

    print('Total number of batches', n_batch_A * n_batch_B)

    # this is a cpu tensor to save space
    # cdist_mat = torch.zeros([n_samples, n_distance])
    k_dist_mat = torch.zeros([n_samples, n_batch_B * k])
    k_inds_mat = torch.zeros([n_samples, n_batch_B * k]).int()

    for i, index_A in enumerate(batch_index_A):
        for j, index_B in enumerate(batch_index_B):
            # get the dist
            cdist_mat_batch = torch.cdist(
                A[index_A[0]:index_A[1], :].to(device),
                B[index_B[0]:index_B[1], :].to(device),
                p=p)

            # important, need to select from the batch index
            # otherwise the ind starts from 0 again
            batch_inds = torch.arange(index_B[0], index_B[1]).repeat(
                batch_size, 1)
            # print(batch_inds.shape)

            bk = bottomk(cdist_mat_batch, k, device=device)
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


def knn_batch(A, B, k=5, p=2.0, batch_size=None, device='cpu'):
    if batch_size is None:
        return knn_full(A, B, k, p, device)
    intermediate_knn = knn_batch_intermediate(A, B, k, p,
                                              batch_size, device)
    return get_knn_from_intermediate(intermediate_knn, k)


def get_cosine_similarity(input1, input2, use_cuda=False):
    # todo: fix use cuda 
    # torch.sum(nn_1* nn_2, dim=1) / (torch.linalg.norm(nn_1, dim=1)**2 * torch.linalg.norm(nn_2, dim=1)**2)
    return torch.sum(input1 * input2, dim=1) / (
            torch.linalg.norm(input1, dim=1) ** 2 *
            torch.linalg.norm(input2, dim=1) ** 2)
