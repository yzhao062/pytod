# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 21:44:34 2021
"""

import numpy as np
import torch

from .basic_operators import cdist_s
from .basic_operators import topk, bottomk, intersec1d
from ..utils.utility import get_batch_index


def cdist_batch(A, B, p=2.0, batch_size=None):
    """Batch version of cdist

    Parameters
    ----------
    A
    B
    p
    batch_size

    Returns
    -------

    """

    # batch is not needed
    if batch_size is None or batch_size >= A.shape[0]:
        return torch.cdist(A.cuda(), B.cuda(), p=p)

    if B is None:
        B = A

    n_samples, n_features = A.shape[0], A.shape[1]
    n_distance = B.shape[0]

    batch_index_A = get_batch_index(n_samples, batch_size)
    batch_index_B = get_batch_index(n_distance, batch_size)
    # print(batch_index_A)
    # print(batch_index_B)

    # this is a cpu tensor to save space
    cdist_mat = torch.zeros([n_samples, n_distance])

    for i, index_A in enumerate(batch_index_A):
        for j, index_B in enumerate(batch_index_B):
            cdist_mat[index_A[0]:index_A[1], index_B[0]:index_B[1]] = \
                cdist_s(A[index_A[0]:index_A[1], :].cuda(),
                        B[index_B[0]:index_B[1], :].cuda()
                        ).cpu()
    return cdist_mat


def topk_batch(A, k, dim=1, batch_size=None):
    if batch_size is None:
        print("original")
        return topk(A.cuda(), k, dim)
    else:
        n_samples = A.shape[0]
        batch_index = get_batch_index(n_samples, batch_size)
        index_mat = torch.zeros([n_samples, k])
        value_mat = torch.zeros([n_samples, k])

        for i, index in enumerate(batch_index):
            print('batch', i)
            tk = topk(A[index[0]:index[1], :].cuda(), k, dim=dim)
            value_mat[index[0]:index[1], :], index_mat[index[0]:index[1], :] = \
                tk[0], tk[1]

        return value_mat, index_mat


def bottomk_batch(A, k, dim=1, batch_size=None):
    # half canm be a choice
    if batch_size is None:
        print("original")
        return bottomk(A.cuda(), k, dim)

    else:
        n_samples = A.shape[0]
        batch_index = get_batch_index(n_samples, batch_size)
        index_mat = torch.zeros([n_samples, k])
        value_mat = torch.zeros([n_samples, k])

        for i, index in enumerate(batch_index):
            print('batch', i)
            tk = bottomk(A[index[0]:index[1], :].cuda(), k, dim=dim)
            value_mat[index[0]:index[1], :], index_mat[index[0]:index[1], :] = \
                tk[0], tk[1]

        return value_mat, index_mat


def intersec1d_batch(t1, t2, batch_size=100000):
    if batch_size >= len(t1) or batch_size >= len(t2):
        return intersec1d(t1, t2)

    batch_index_A = get_batch_index(len(t1), batch_size)
    batch_index_B = get_batch_index(len(t2), batch_size)
    print(batch_index_A)
    print(batch_index_B)

    # use cuda for fast computation
    candidate_set = torch.tensor([]).cuda()
    for i, index_A in enumerate(batch_index_A):
        for j, index_B in enumerate(batch_index_B):
            candidate_set = torch.cat((candidate_set,
                                       intersec1d(t1[index_A[0]:index_A[1]],
                                                  t2[index_B[0]:index_B[1]])),
                                      dim=0)
    return torch.unique(candidate_set).cpu()
