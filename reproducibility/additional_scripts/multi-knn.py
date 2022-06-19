# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 14:24:54 2021

@author: yuezh
"""
import time
import torch
import torch.multiprocessing as mp
import numpy as np

from basic_operators import bottomk
# from basic_operators_batch import cdist_batch
from basic_operators_batch import bottomk_batch
from utility import get_batch_index

def bottomk(A, k, dim=1):
    if len(A.shape) == 1:
        dim = 0
    # tk = torch.topk(A * -1, k, dim=dim)
    # see parameter https://pytorch.org/docs/stable/generated/torch.topk.html
    tk = torch.topk(A, k, dim=dim, largest=False)
    return tk[0].cpu(), tk[1].cpu()

# https://discuss.pytorch.org/t/using-torch-tensor-over-multiprocessing-queue-process-fails/2847/12




def cdist_batch_k(A, B, p=2.0, gpu_id=0, batch_size=None):
    # def cdist_batch(A, B=None, batch_size=None):
    # TODO: whether to half can be a parameter
    # TODO: should pass other possible hyperparameters to torch.cdist

    if B is None:
        B = A

    n_samples, n_features = A.shape[0], A.shape[1]
    n_distance = B.shape[0]

    # batch is not needed
    if batch_size is None or batch_size >= n_samples:
        print('direct cal')
        kd, ki = bottomk(torch.cdist(A.to(gpu_id), B.to(gpu_id), p=p), k=10)
        return kd, ki

    batch_index_A = get_batch_index(n_samples, batch_size)
    batch_index_B = get_batch_index(n_distance, batch_size)
    print(batch_index_A)
    print(batch_index_B)

    # this is a cpu tensor to save space
    # cdist_mat = torch.zeros([n_samples, n_distance])
    
    # print(gpu_id, gpu_id, gpu_id, gpu_id)
    kd_list = []
    for i, index_A in enumerate(batch_index_A):
        for j, index_B in enumerate(batch_index_B):
            for t in range(3):
                kd, ki = bottomk(torch.cdist(A[index_A[0]:index_A[1], :].to(gpu_id),
                                              B[index_B[0]:index_B[1], :].to(gpu_id), p=p), 10)
            
            # kd, ki = bottomk(torch.cdist(A[index_A[0]:index_A[1], :].cuda(gpu_id),
            #                               B[index_B[0]:index_B[1], :].cuda(gpu_id), p=p),
            #                   10)
            # kd, ki = bottomk(torch.cdist(A[index_A[0]:index_A[1], :].cuda(gpu_id),
            #                               B[index_B[0]:index_B[1], :].cuda(gpu_id), p=p),
            #                   10)
            
            # kd, ki = bottomk(torch.cdist(A[index_A[0]:index_A[1], :].cuda(gpu_id),
            #                               B[index_B[0]:index_B[1], :].cuda(gpu_id), p=p),
            #                   10)
            
            # kd, ki = bottomk(torch.cdist(A[index_A[0]:index_A[1], :].cuda(gpu_id),
            #                               B[index_B[0]:index_B[1], :].cuda(gpu_id), p=p),
            #                   10)
            kd_list.append((kd, ki))
            
            # cdist_mat[index_A[0]:index_A[1], index_B[0]:index_B[1]] = \
            #     torch.cdist(A[index_A[0]:index_A[1], :].cuda(gpu_id),
            #                 B[index_B[0]:index_B[1], :].cuda(gpu_id),
            #                 p=p).cpu()
    # print(gpu_id, kd_list)
    # return cdist_mat
    return kd_list 

    
    # return bottomk_batch(cdist_mat, k=10, batch_size=batch_size)
    
    

def cdist_per_GPU(x, pval, list_indexs, k, gpu_id, GPU_batch):
    
    print('something')
    print(list_indexs)
    for i in list_indexs:
        # print(gpu_id, i)
        print('On GPU', gpu_id, i[0][0], i[0][1], i[1][0], i[1][1])
        
        # # a = x[i[0][0]:i[0][1], :].to(gpu_id)
        # # batch_size = i[1][1] - i[1][0]
        # # batch_inds = torch.arange(i[1][0], i[1][1]).repeat(batch_size, 1)
        kd_list = cdist_batch_k(x[i[0][0]:i[0][1], :],
                                        x[i[1][0]:i[1][1], :],
                                        gpu_id=gpu_id,
                                        batch_size=GPU_batch)
        
        # kd_list = knn_batch_gpu(x[i[0][0]:i[0][1], :],
        #                                 x[i[1][0]:i[1][1], :],
        #                                 gpu_id=gpu_id,
        #                                 batch_size=GPU_batch)
        
        # print('cdist_mat_batch shape', cdist_mat_batch.shape)
        
        # bk = bottomk_batch(cdist_mat_batch, k, batch_size=GPU_batch)
        
        # bk = batch_cdist(x[i[0][0]:i[0][1], :],  
        #                  x[i[1][0]:i[1][1], :],
        #                  batch_size=GPU_batch)
        # bk = bottomk(cdist_mat_batch, k)
        
        # print('bk!', bk[0].shape, bk[1].shape)
        # print('b ind', batch_inds.shape)
        # print('aa', batch_inds.gather(1, bk[1].long()).shape)
        # pval.append((i, bk[0], batch_inds.gather(1, bk[1].long())))

        # pval.append((i, batch_cdist(x[i[0][0]:i[0][1], :],  
        #                             x[i[1][0]:i[1][1], :],
        #                             batch_size=GPU_batch).cpu()))
        
        pval.append((gpu_id, kd_list))
    
if __name__ == '__main__':
    mp.set_start_method('spawn')
    
    # data generation
    n_processes = 8    
    n_samples =4000000
    n_dimensions = 100
    k =10
    
    # decide global and local size
    global_batch_size= 250000
    GPU_batch = 50000
    
    X = torch.randn(n_samples,n_dimensions)
    batch_index_A = get_batch_index(n_samples, global_batch_size)
    
    # retrieve all index
    all_index = []
    for i in batch_index_A:
        for j in batch_index_A:
            all_index.append((i, j))
    n_tasks_per_gpu = int(len(all_index)/n_processes)
    
    ab = []
    
    start = time.time()
    
    with mp.Manager() as mgr:
        processes = []    
        pval = mgr.list()

        for i in range(n_processes):
            processes.append(mp.Process(target=cdist_per_GPU, 
                                        args=(X, 
                                              pval, 
                                              all_index[i*n_tasks_per_gpu:(i+1)*n_tasks_per_gpu],
                                              k,
                                              i,
                                              GPU_batch)))
        for p in processes:
            p.start()
    
        for p in processes:
            p.join()
        
        print()
        print()
        print('cdist from {a} gpus'.format(a=n_processes))
        for k, pv in enumerate(pval):
            print(k, len(pv))
            ab.append(pv)

        
    kdist_mat = torch.zeros([n_samples, int(n_samples/GPU_batch)*10])
    kind_mat = torch.zeros([n_samples, int(n_samples/GPU_batch)*10])
    for k, pv in enumerate(ab):
        index_left = all_index[pv[0]][0]
        index_right = all_index[pv[0]][1]
        
        start_index = int(index_right[0]/GPU_batch)
        sub_index_len = len(pv[1])
        
        # sub_mat = torch.zeros(global_batch_size*global_batch_size*GPU_batch, 10)
        # print('submat', sub_mat.shape)
        single_len = int(np.sqrt(sub_index_len))
        
        for i in range(single_len):
            
            if index_right[0] !=0:
                for j in range(single_len, 2*single_len):
                    print(index_left[0], index_left[0]+(i+1)*GPU_batch, 10*j, 10*(j+1))
                    kdist_mat[index_left[0]+i*GPU_batch:index_left[0]+(i+1)*GPU_batch, 10*j:10*(j+1)] = pv[1][i*single_len+j-single_len][0]
                    kind_mat[index_left[0]+i*GPU_batch:index_left[0]+(i+1)*GPU_batch, 10*j:10*(j+1)] = pv[1][i*single_len+j-single_len][1]
            else:
                for j in range(single_len):
                    print(index_left[0], index_left[0]+(i+1)*GPU_batch, 10*j, 10*(j+1))
                    kdist_mat[index_left[0]+i*GPU_batch:index_left[0]+(i+1)*GPU_batch, 10*j:10*(j+1)] = pv[1][i*single_len+j][0]
                    kind_mat[index_left[0]+i*GPU_batch:index_left[0]+(i+1)*GPU_batch, 10*j:10*(j+1)] = pv[1][i*single_len+j][1]
        # for l in range(sub_index_len):
        #     print(l*GPU_batch, (l+1)*GPU_batch)
        #     print('s', pv[1][l][0].shape)
        #     # sub_mat[l*GPU_batch:(l+1)*GPU_batch, :] = pv[1][l][0]
        # print(sub_mat.shape)
    print(kdist_mat, kdist_mat.shape)
    
    knn_dist, knn_inds = bottomk(kdist_mat.cuda(), k=10)
    # need a final gather since kind_mat is for specific mat.

    end = time.time()
    print(end - start)
        
