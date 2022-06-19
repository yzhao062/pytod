# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

import os
import sys
import time
import itertools
import torch

# temporary solution for relative imports in case pyod is not installed
# if pyod is installed, no need to use the following line
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname("__file__"), '..')))
# supress warnings for clean output
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.io import loadmat

from pyod.models.abod import ABOD
from pyod.models.cblof import CBLOF
# from pyod.models.feature_bagging import FeatureBagging
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lmdd import LMDD
from pyod.models.loci import LOCI
from pyod.models.loda import LODA
from pyod.models.lof import LOF
from pyod.models.mcd import MCD
from pyod.models.ocsvm import OCSVM
from pyod.models.pca import PCA
from pyod.models.cof import COF
from pyod.models.sod import SOD

from pyod.utils.data import generate_data
from pyod.utils.data import evaluate_print

from pyod.utils.utility import standardizer
from pyod.utils.utility import precision_n_scores
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import MinMaxScaler
import arff

from scipy.stats import rankdata
from basic_operators import topk, bottomk, bottomk_low_prec, topk_low_prec
from basic_operators_batch import cdist_batch, topk_batch, bottomk_batch, intersec1d_batch
from basic_operators import topk, intersec1d
from intermediate_layers import neighbor_within_range, \
    neighbor_within_range_low_prec, knn_batch
from intermediate_layers import neighbor_within_range_low_prec_float, get_bounded_error, get_indices_clear_pairs

from utility import Standardizer, get_batch_index

from pytorch_memlab import LineProfiler
from pytorch_memlab import MemReporter

#%%

# knn

# 原始数据存在cpu和memory中，在batch的时候call cuda即可
# 在极端情况下，cpu的memory也会被塞满
A = torch.randn(100000, 1000)
B = torch.randn(100000, 1000)
k=5

# A = torch.from_numpy(C)
# B = torch.from_numpy(D)
# with LineProfiler(knn_batch) as prof:
#     start = time.time()
#     a,b = knn_batch(A, B, k, batch_size=20000)
#     end = time.time()
#     print(end - start)
# print(prof.display())

# with LineProfiler(knn_full) as prof:
#     start = time.time()
#     c, d = knn_full(A, B, k)
#     end = time.time()
#     print(end - start)
# print(prof.display())

# from sklearn.neighbors import NearestNeighbors
# C = np.random.randn(100000, 1000)
# D = np.random.randn(100000, 1000)
# start = time.time()
# clf = NearestNeighbors(n_neighbors=k)
# clf.fit(C)
# e,f = clf.kneighbors(D, return_distance=True)
# end = time.time()
# print(end - start)

#%% neighboir within range 
n_train = 20000  # number of training points
n_features = 100

# Generate sample data
# X_train = torch.randn([n_train, n_features]).half()
# X_train = torch.randn([n_train, n_features])
X_train = torch.randn([n_train, n_features]).double()
X_train_norm = Standardizer(X_train, return_mean_std=False)

# with LineProfiler(neighbor_within_range_low_prec) as prof:
#     start = time.time()
#     clear_pairs_low_prec = neighbor_within_range_low_prec(X_train_norm, range_threshold=12)
#     end = time.time()
#     print(end - start)
# print(prof.display())

# with LineProfiler(neighbor_within_range_low_prec_float) as prof:
#     start = time.time()
#     clear_pairs_low_prec = neighbor_within_range_low_prec_float(X_train_norm, range_threshold=12)
#     end = time.time()
#     print(end - start)
# print(prof.display())


with LineProfiler(neighbor_within_range) as prof:
    start = time.time()
    clear_pairs = neighbor_within_range(X_train_norm, range_threshold=12)
    end = time.time()
    print(end - start)
print(prof.display())


# # return the neighbors indices for each sample
# a = get_indices_clear_pairs(clear_pairs, 0)
# a_low = get_indices_clear_pairs(clear_pairs_low_prec, 0)
# assert (a.shape == a_low.shape)



#%%
contamination = 0.1  # percentage of outliers
n_train = 200000  # number of training points
n_test = 1000  # number of testing points
n_features = 2000
    
# Generate sample data
# X_train = torch.randn([n_train, n_features]).half()
# X_train = torch.randn([n_train, n_features])
X_train = torch.randn([n_train, n_features]).double()

k = 5

with LineProfiler(topk_low_prec) as prof:
    start = time.time()
    bottomk_dist, bottomk_indices = topk_low_prec(X_train, k)
    # bottomk_dist, bottomk_indices = topk(X_train.cuda(), k)
    end = time.time()
    print(end - start)
print(prof.display())


#%%
import torch
import torch.autograd.profiler as profiler
# t1 = torch.randint(low=0, high=10000000, size=[150000000])
# t2 = torch.randint(low=5000000, high=20000000, size=[150000000])

t1 = torch.randint(low=0, high=1000000, size=[100000])
t2 = torch.randint(low=1000000, high=2000000, size=[100000])

# https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html
with profiler.profile(profile_memory=True, record_shapes=True) as prof:
    with profiler.record_function("intersection"):
# with LineProfiler(intersec1d) as prof:
        start = time.time()
        # w = intersec1d(t1.cuda(), t2.cuda())
        w = intersec1d_batch(t1.cuda(), t2.cuda(), batch_size=50000)
        end = time.time()
        print(end - start)
# print(prof.display())

# with LineProfiler(intersec1d_batch) as prof:
#     start = time.time()
#     w = intersec1d_batch(t1.cuda(), t2.cuda(), batch_size=500000)
#     end = time.time()
#     print(end - start)
# print(prof.display())


#%%
C = np.random.randn(1000000)
D = np.random.randn(1000000)
start = time.time()
w = np.intersect1d(C,D)
end = time.time()
print(end - start)
#%%

# A = torch.randn(10000000, 1000)
# # A = torch.randn(1000, 100)
# k = 10 

# # 原始版本会爆显存

# with LineProfiler(topk_batch) as prof:
#     start = time.time()
#     v1, ind1 = topk_batch(A, k, batch_size=1000000)
#     end = time.time()
#     print(end - start)
# print(prof.display())

# with LineProfiler(topk) as prof:
#     start = time.time()
#     v1, ind1 = topk(A, k)
#     end = time.time()
#     print(end - start)
# print(prof.display())


# start = time.time()
# v, ind = bottomk_batch(A,k)
# end = time.time()
# print(end - start)

# print()
# start = time.time()
# v1, ind1 = bottomk_batch(A,k, batch_size=1000000)
# end = time.time()
# print(end - start)


# numpy version没有exact topk的方法，需要自己搞
# 网络实现
def topk_(matrix, K, axis=1):
    if axis == 0:
        row_index = np.arange(matrix.shape[1 - axis])
        topk_index = np.argpartition(-matrix, K, axis=axis)[0:K, :]
        topk_data = matrix[topk_index, row_index]
        topk_index_sort = np.argsort(-topk_data,axis=axis)
        topk_data_sort = topk_data[topk_index_sort,row_index]
        topk_index_sort = topk_index[0:K,:][topk_index_sort,row_index]
    else:
        column_index = np.arange(matrix.shape[1 - axis])[:, None]
        topk_index = np.argpartition(-matrix, K, axis=axis)[:, 0:K]
        topk_data = matrix[column_index, topk_index]
        topk_index_sort = np.argsort(-topk_data, axis=axis)
        topk_data_sort = topk_data[column_index, topk_index_sort]
        topk_index_sort = topk_index[:,0:K][column_index,topk_index_sort]
    return topk_data_sort, topk_index_sort

C = np.random.randn(10000000, 1000)
print()
start = time.time()
v1, ind1 = topk_(C, K=10)
end = time.time()
print(end - start)

#%%

# 原始数据存在cpu和memory中，在batch的时候call cuda即可
# 在极端情况下，cpu的memory也会被塞满
A = torch.randn(50000, 1000)
B = torch.randn(50000, 1000)


with LineProfiler(cdist_batch) as prof:
    start = time.time()
    w = cdist_batch(A, B, batch_size=5000)
    end = time.time()
    print(end - start)
print(prof.display())

# with LineProfiler(torch.cdist) as prof:
#     start = time.time()
#     w = torch.cdist(A.cuda(), B.cuda()) 
#     end = time.time()
#     print(end - start)
# print(prof.display())

# start = time.time()
# # w = batch_cdist(A, B, batch_size=5000)
# w = torch.cdist(A.cuda(), B.cuda()) 
# end = time.time()
# print(end - start)

# torch.cuda.empty_cache()

# # GPU版本在50000的时候就没法运行了。
# start = time.time()
# cdist_mat_raw = cdist(A.cuda(), B.cuda())
# # print(cdist_mat_raw)
# end = time.time()
# print(end - start)
#%% numpy time for cdist
import time
import numpy as np
from scipy.spatial.distance import cdist
C = np.random.randn(50000, 1000)
D = np.random.randn(50000, 1000)
start = time.time()
cdist_mat_raw = cdist(C, D)
end = time.time()
print(end - start)

