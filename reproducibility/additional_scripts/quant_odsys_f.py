import numpy as np
import torch
from mpmath import mp, mpf
from pyod.utils.utility import get_list_diff
from basic_operators import topk#, topk_low_prec
from utility_memory import get_max_active_cuda

def get_bounded_error(max_value, dimension, machine_eps=np.finfo(float).eps, two_sided=True):
    mp.dps = 100
    factor = (1+machine_eps)**(mp.log(dimension)+2)-1
    if two_sided:
        return float(2*(4*dimension*(max_value**2)*factor))
    else:
        return float(4*dimension*(max_value**2)*factor)

def neighbor_within_range(X, range_threshold):
    # calculate the cdist in lower precision
    distance_mat = torch.cdist(X.cuda(), X.cuda()).cpu()
    # print(distance_mat)
    # identify the indice pairs
    clear_indices = torch.nonzero((distance_mat<= range_threshold), 
                                  as_tuple=False)
    return clear_indices

def neighbor_within_range_low_prec(X, range_threshold):
    
    n_samples, n_features = X.shape[0], X.shape[1]
    
    # get the error bound
    error_bound = float(get_bounded_error(torch.max(X).cpu().numpy(), n_features))
    
    # calculate the cdist in lower precision
    distance_mat = torch.cdist(X.cuda().half(), X.cuda().half()).cpu()
    # selected_indice = torch.nonzero(distance_mat<= threshold, as_tuple=False)
    
    # we can calculate diffence instead
    # see code here
    full_indices = np.arange(0, n_samples)
    
    # identify the ambiguous indice pairs
    amb_indices = torch.nonzero((distance_mat<= range_threshold+error_bound) & 
                                (distance_mat>= range_threshold-error_bound), as_tuple=False)
    
    print(amb_indices.shape)
    print(torch.unique(amb_indices[:, 0]).shape)
    
    
    # these are the indices of the samples with no ambiguity
    clear_indices = get_list_diff(full_indices, amb_indices[:, 0].cpu().numpy())
    
    
    # find the matched samples 
    clear_pairs = torch.nonzero(distance_mat[clear_indices, :]<= range_threshold, as_tuple=False)
    print('initial pairs', clear_pairs.shape)
    
    # recalculate for the amb_indices
    # get the samples for recalculation
    amb_1 = X[amb_indices[:, 0], :]
    amb_2 = X[amb_indices[:, 1], :]
    
    pdist = torch.nn.PairwiseDistance(p=2)
    amb_dist = pdist(amb_1, amb_2)
    
    # finally return a 2-d tensor containing all the tensors
    true_neigh_indices = torch.nonzero((amb_dist<= range_threshold), as_tuple=True)
    # print(true_neigh_indices[0].shape)
    
    
    clear_pairs = torch.cat((clear_pairs, amb_indices[true_neigh_indices[0], :]))
    print('ultimate true pairs', clear_pairs.shape)
    
    print('imprecision pairs correct:', amb_indices.shape[0]-true_neigh_indices[0].shape[0])
    return clear_pairs

def neighbor_within_range_low_prec_float(X, range_threshold):
    
    n_samples, n_features = X.shape[0], X.shape[1]
    
    # get the error bound
    error_bound = float(get_bounded_error(torch.max(X).cpu().numpy(), n_features))
    
    # calculate the cdist in lower precision
    distance_mat = torch.cdist(X.cuda().float(), X.cuda().float()).cpu()
    
    # selected_indice = torch.nonzero(distance_mat<= threshold, as_tuple=False)
    
    # we can calculate diffence instead
    # see code here
    full_indices = np.arange(0, n_samples)
    
    # identify the ambiguous indice pairs
    amb_indices = torch.nonzero((distance_mat<= range_threshold+error_bound) & 
                                (distance_mat>= range_threshold-error_bound), as_tuple=False)
    
    print(amb_indices.shape)
    print(torch.unique(amb_indices[:, 0]).shape)
    
    
    # these are the indices of the samples with no ambiguity
    clear_indices = get_list_diff(full_indices, amb_indices[:, 0].cpu().numpy())
    
    
    # find the matched samples 
    clear_pairs = torch.nonzero(distance_mat[clear_indices, :]<= range_threshold, as_tuple=False)
    print('initial pairs', clear_pairs.shape)
    
    # recalculate for the amb_indices
    # get the samples for recalculation
    amb_1 = X[amb_indices[:, 0], :]
    amb_2 = X[amb_indices[:, 1], :]
    
    pdist = torch.nn.PairwiseDistance(p=2)
    amb_dist = pdist(amb_1, amb_2)
    
    # finally return a 2-d tensor containing all the tensors
    true_neigh_indices = torch.nonzero((amb_dist<= range_threshold), as_tuple=True)
    # print(true_neigh_indices[0].shape)
    
    
    clear_pairs = torch.cat((clear_pairs, amb_indices[true_neigh_indices[0], :]))
    print('ultimate true pairs', clear_pairs.shape)
    
    print('imprecision pairs correct:', amb_indices.shape[0]-true_neigh_indices[0].shape[0])
    return clear_pairs

def Standardizer(X_train, mean=None, std=None, return_mean_std=False):
    
    if mean is None:
        mean = torch.mean(X_train, axis=0)
        std = torch.std(X_train, axis=0)
        # print(mean.shape, std.shape)
        assert (mean.shape[0] == X_train.shape[1])
        assert (std.shape[0] == X_train.shape[1])
    
    
    X_train_norm = (X_train-mean)/std
    assert(X_train_norm.shape == X_train.shape)
    
    if return_mean_std:
        return X_train_norm, mean, std
    else:
        return  X_train_norm
    

import os
import torch
from pytorch_memlab import LineProfiler
import time
from scipy.io import loadmat

n_iter = 10


# mat_file = 'cifar-10'
# loaded = np.load(os.path.join('datasets', 'cifar', 'cifar-10.npz'))
# X = loaded['X']
# y = loaded['y']
# y[y!=1] = 0

# X_torch = torch.from_numpy(X).double()
# X_train_norm = Standardizer(X_torch, return_mean_std=False)

# mat_file = "speech.mat"
# # loading and vectorization
# mat = loadmat(os.path.join("datasets", "ODDS", mat_file))

# X = mat['X'].astype('float')
# y = mat['y'].ravel()
# X_train = torch.from_numpy(X).double()
# X_train_norm = Standardizer(X_train, return_mean_std=False)

# Generate sample data
n_train = 3686  # number of training points
n_features = 400
# X_train = torch.randn([n_train, n_features]).half()
# X_train = torch.randn([n_train, n_features])
X_train = torch.randn([n_train, n_features]).double()
X_train_norm = Standardizer(X_train, return_mean_std=False)




def topk_low_prec(A, k, dim=1, mode='half', sort_value=False):
    
    # in lower precision
    if mode=='half':
        # do conversion first
        A_GPU = A.half().cuda()

    else:
        A_GPU = A.float().cuda()
        
    # print(A_GPU)
    
    topk_dist, topk_indices = topk(A_GPU, k+1)
    
    # topk(A, k+1)
    # print(A)
    # get all the ambiguous indices with 2-element assumption
    amb_indices_p1 = torch.where(topk_dist[:, k] >= topk_dist[:, k-1])[0]
    amb_indices_m1 = torch.where(topk_dist[:, k-2] <= topk_dist[:, k-1])[0]
    
    # there might be repetition, so we need to find the unique element only
    amb_indices = torch.unique(torch.cat((amb_indices_p1, amb_indices_m1)))
    
    print("ambiguous indices", len(amb_indices))
    
    
    A_GPU_recal = A[amb_indices, :].cuda()
    # recal_cdist = cdist_dist[amb_indices, :].double()
    _, topk_indices[amb_indices, :k] = topk(A_GPU_recal, k)
    
    # drop the last bit k+1
    topk_indices = topk_indices[:, :k].cpu()
    
    # select by indices for bottom distance
    # https://stackoverflow.com/questions/58523290/select-mask-different-column-index-in-every-row
    
    topk_dist = A.gather(1, topk_indices)
    
    if sort_value:
        topk_dist_sorted, topk_indices_argsort = torch.sort(topk_dist, dim=dim, descending=True)
        topk_indices_sorted = topk_indices.gather(1, topk_indices_argsort)
        return topk_dist_sorted, topk_indices_sorted
    else:
        return topk_dist, topk_indices

k = 20
print("16-bit precision")
temp_result = []
with LineProfiler(topk_low_prec) as prof:
    for i in range(n_iter):
        start = time.time()
        k_dist, k_ind = topk_low_prec(X_train_norm, k=k, mode='half')
        end = time.time()
        print(end - start)
        temp_result.append(end - start)
print(prof.display())
print("16-bit precision", np.mean(temp_result[1:]), get_max_active_cuda(prof), 'mb')

print("32-bit precision")
temp_result = []
with LineProfiler(topk_low_prec) as prof:
    for i in range(n_iter):
        start = time.time()
        k_dist, k_ind = topk_low_prec(X_train_norm, k=k, mode='float')
        end = time.time()
        print(end - start)
        temp_result.append(end - start)
print(prof.display())
print("32-bit precision", np.mean(temp_result[1:]), get_max_active_cuda(prof), 'mb')

print("64-bit precision")
temp_result = []
with LineProfiler(topk) as prof:
    for i in range(n_iter):
        start = time.time()
        k_dist, k_ind = topk(X_train_norm, k=k)
        end = time.time()
        print(end - start)
        temp_result.append(end - start)
print(prof.display())
print("64-bit precision", np.mean(temp_result[1:]), get_max_active_cuda(prof), 'mb')

# # Generate sample data
# n_train = 3062  # number of training points
# n_features = 166
# # X_train = torch.randn([n_train, n_features]).half()
# # X_train = torch.randn([n_train, n_features])
# X_train = torch.randn([n_train, n_features]).double()
# X_train_norm = Standardizer(X_train, return_mean_std=False)

# print("16-bit precision")
# temp_result = []
# with LineProfiler(neighbor_within_range_low_prec) as prof:
#     for i in range(n_iter):
#         start = time.time()
#         clear_pairs_low_prec = neighbor_within_range_low_prec(X_train_norm, range_threshold=16)
#         end = time.time()
#         print(end - start)
#         temp_result.append(end - start)
# print(prof.display())
# print("16-bit precision", np.mean(temp_result[1:]), get_max_active_cuda(prof), 'mb')

# print("32-bit precision")
# temp_result = []
# with LineProfiler(neighbor_within_range_low_prec_float) as prof:
#     for i in range(n_iter):
#         start = time.time()
#         clear_pairs_low_prec = neighbor_within_range_low_prec_float(X_train_norm, range_threshold=16)
#         end = time.time()
#         print(end - start)
#         temp_result.append(end - start)
# print(prof.display())
# print("32-bit precision", np.mean(temp_result[1:]), get_max_active_cuda(prof), 'mb')

# print("64-bit precision")
# temp_result = []
# with LineProfiler(neighbor_within_range) as prof:
#     for i in range(n_iter):
#         start = time.time()
#         clear_pairs_low_prec = neighbor_within_range(X_train_norm, range_threshold=16)
#         end = time.time()
#         print(end - start)
#         temp_result.append(end - start)
# print(prof.display())
# print("64-bit precision", np.mean(temp_result[1:]), get_max_active_cuda(prof), 'mb')