# -*- coding: utf-8 -*-
"""
Created on Sat Apr  3 14:37:27 2021

@author: yuezh
"""
import time
import numpy as np
import torch

from basic_operators import bottomk, cdist
from utility import get_batch_index, Standardizer


def batch_cdist(A, B, p=2.0, batch_size=None):
    # def cdist_batch(A, B=None, batch_size=None):
    # TODO: whether to half can be a parameter
    # TODO: should pass other possible hyperparameters to torch.cdist
    
    # batch is not needed
    if batch_size is None:
       return torch.cdist(A.cuda(), B.cuda(), p=p) 
    #todo: what if n_samples is smaller than batch size. need an if/else check

    if B is None:
        B = A

    n_samples, n_features = A.shape[0], A.shape[1]
    n_distance = B.shape[0]

    batch_index_A = get_batch_index(n_samples, batch_size)
    batch_index_B = get_batch_index(n_distance, batch_size)
    print(batch_index_A)
    print(batch_index_B)

    # this is a cpu tensor to save space
    cdist_mat = torch.zeros([n_samples, n_distance])

    for i, index_A in enumerate(batch_index_A):
        for j, index_B in enumerate(batch_index_B):
            cdist_mat[index_A[0]:index_A[1], index_B[0]:index_B[1]] = \
                torch.cdist(A[index_A[0]:index_A[1], :].cuda(),
                            B[index_B[0]:index_B[1], :].cuda(),
                            p=p).cpu()
    return cdist_mat

def knn_batch_intermediate(A, B, k=5, p=2.0, batch_size=None):
    # this is the map step
    n_samples, n_features = A.shape[0], A.shape[1]
    n_distance = B.shape[0]
    
    batch_index_A = get_batch_index(n_samples, batch_size)
    batch_index_B = get_batch_index(n_distance, batch_size)
    print(batch_index_A)
    print(batch_index_B)
    
    n_batch_A = len(batch_index_A)
    n_batch_B = len(batch_index_B)
    
    # this is a cpu tensor to save space
    # cdist_mat = torch.zeros([n_samples, n_distance])
    k_dist_mat = torch.zeros([n_samples, n_batch_B*k])
    k_inds_mat = torch.zeros([n_samples, n_batch_B*k]).int()
    
    for i, index_A in enumerate(batch_index_A):
        for j, index_B in enumerate(batch_index_B):
            print(i, j, n_batch_A, n_batch_B)
            
            # get the dist
            cdist_mat_batch = torch.cdist(A[index_A[0]:index_A[1], :].cuda(),
                                          B[index_B[0]:index_B[1], :].cuda(), p=p)
            
            # important, need to select from the batch index
            # otherwise the ind starts from 0 again
            batch_inds = torch.arange(index_B[0], index_B[1]).repeat(batch_size, 1)
            # print(batch_inds.shape)
            
            bk = bottomk(cdist_mat_batch, k)
            # we need a global indices here
            k_dist_mat[i*batch_size:(i+1)*batch_size, j*k:(j+1)*k] = bk[0]
            k_inds_mat[i*batch_size:(i+1)*batch_size, j*k:(j+1)*k] = batch_inds.gather(1, bk[1].long())
    
    return k_dist_mat, k_inds_mat


def get_knn_from_intermediate(intermediate_knn, k):
    # this is the reduce step
    
    # sort distance for index, real knn happens here
    sorted_ind = torch.argsort(intermediate_knn[0], dim=1) 
    
    # bottomk_indices.gather(1, bottomk_indices_argsort)
    
    # selected the first k for each sample
    knn_dist = intermediate_knn[0].gather(1, sorted_ind[:, :k])
    knn_inds = intermediate_knn[1].gather(1, sorted_ind[:, :k])
    
    return knn_dist, knn_inds

def knn_batch(A, B, k=5, p=2.0, batch_size=None):
    intermediate_knn = knn_batch_intermediate(A, B, k, p, batch_size)
    return get_knn_from_intermediate(intermediate_knn, k)


if __name__ == '__main__':

    n_train = 1000000  # number of training points
    n_features = 100
    batch_size = 40000
    p = 2
    k = 10
    
    # # Generate sample data
    # # X_train = torch.randn([n_train, n_features]).half()
    # # X_train = torch.randn([n_train, n_features])
    A = torch.randn([n_train, n_features])
    # # X_train_norm = Standardizer(X_train, return_mean_std=False)
    B = A
    
    
    
    start = time.time()
    # intermediate_knn = knn_batch_intermediate(A, B, k, batch_size=batch_size)
    # knn_dist, knn_inds = get_knn_from_intermediate(intermediate_knn, k)
    
    knn_dist, knn_inds = knn_batch(A, B, k, batch_size=batch_size)
    
    end = time.time()
    print(n_train, n_features, end - start)

