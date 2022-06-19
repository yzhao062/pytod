# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 18:11:07 2020

@author: yuezh
"""
gpu_info = !nvidia-smi
gpu_info = '\n'.join(gpu_info)
if gpu_info.find('failed') >= 0:
  print('Select the Runtime > "Change runtime type" menu to enable a GPU accelerator, ')
  print('and then re-execute this cell.')
else:
  print(gpu_info)

from psutil import virtual_memory
ram_gb = virtual_memory().total / 1e9
print('Your runtime has {:.1f} gigabytes of available RAM\n'.format(ram_gb))

if ram_gb < 20:
  print('To enable a high-RAM runtime, select the Runtime > "Change runtime type"')
  print('menu, and then select High-RAM in the Runtime shape dropdown. Then, ')
  print('re-execute this cell.')
else:
  print('You are using a high-RAM runtime!')
  

import time
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import cdist
from torch.autograd import Variable
from torchvision.transforms import Normalize

from pyod.utils.data import generate_data
from pyod.utils.data import evaluate_print

from itertools import combinations
# check torch version

print(torch.__version__)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device, torch.cuda.get_device_name(torch.cuda.current_device()))

# disable autograd
torch.set_grad_enabled(False)

def get_batch_index(n_samples, batch_size):
    index_tracker = []

    n_batches = int(np.ceil(n_samples//batch_size))
    print('n_batches', n_batches) 
    tracker = 0
    for i in range(n_batches):
        left_index = tracker*batch_size
        right_index = left_index + batch_size
        tracker += 1
        # print(left_index, right_index)
        index_tracker.append((left_index, right_index))
  
    if n_samples % batch_size != 0:
      left_index = right_index
      right_index = n_samples
      # print(left_index, right_index)
      index_tracker.append((left_index, right_index))
    return index_tracker

#%%
torch.cuda.empty_cache()

A = torch.randn(50000, 1000).cuda().half()
B = torch.randn(50000, 1000).cuda().half()


def batch_cdist(A, B, batch_size = 25000):
    # def cdist_batch(A, B=None, batch_size=None):
    
    # A should be able to be batchfied 
    # B should be fixed
    
    if B is None:
        B = A
    
    n_samples, n_features = A.shape[0], A.shape[1]
    n_distance = B.shape[0]
    
    batch_index_A = get_batch_index(n_samples, batch_size)
    batch_index_B = get_batch_index(n_distance, batch_size)
    print(batch_index_A)
    print(batch_index_B)
    
    cdist_mat = torch.zeros([n_samples, n_distance]).half()
    # cdist_mat = np.zeros([n_samples, n_distance])

    for i, index_A in enumerate(batch_index_A):
        for j, index_B in enumerate(batch_index_B):
            cdist_mat[index_A[0]:index_A[1], index_B[0]:index_B[1]] = \
            cdist(A[index_A[0]:index_A[1], :].cuda().half(), 
                  B[index_B[0]:index_B[1], :].cuda().half())

    print(cdist_mat)
    print()
    return cdist_mat

start = time.time()
w = batch_cdist(A, B)
end = time.time()
print(end - start)

torch.cuda.empty_cache()

start = time.time()
cdist_mat_raw = cdist(A, B)
print(cdist_mat_raw)
end = time.time()
print(end - start)
#%% numpy time
from scipy.spatial.distance import cdist
C = torch.randn(10000, 1000).half().cpu().numpy()
D = torch.randn(10000, 1000).half().cpu().numpy()
start = time.time()
cdist_mat_raw = cdist(C, D)
end = time.time()
print(end - start)

C = torch.randn(10, 2).cuda().half()
print(C)
print(torch.norm(C, dim=1))


#%%
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import cdist
from torch.autograd import Variable
from torchvision.transforms import Normalize

from pyod.utils.data import generate_data
from pyod.utils.data import evaluate_print

from itertools import combinations

torch.cuda.empty_cache()

a = torch.randn(50000, 200).cuda().half()
b = torch.randn(20000, 200).cuda().half()

def cdist_s(a,b):
    norm_a = torch.norm(a, dim=1).reshape(a.shape[0], 1)
    norm_b = torch.norm(b, dim=1).reshape(1, b.shape[0])
    
    w = norm_a**2 + norm_b**2 - 2* torch.matmul(a, b.T)
    return torch.sqrt(w)

print(a)
print(b)

w = cdist_s(a,b)
print(w)
# p = cdist(a,b)
# print(p)
