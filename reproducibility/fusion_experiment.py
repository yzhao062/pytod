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
from pytod.models.basic_operators_batch import cdist_batch
from pytod.models.basic_operators_batch import bottomk_batch

from pytod.models.intermediate_layers import knn_batch

n_train = 150000  # number of training points
n_features = 200
batch_size = 20000
k = 10
# Generate sample data
X_train = torch.randn([n_train, n_features]).float()

device = validate_device(0)

def simple_conct(X_train, batch_size, k, device):
    cdist_dist = cdist_batch(X_train, X_train, batch_size=batch_size,
                             device=device)
    bottomk_batch(cdist_dist, k=k, batch_size=batch_size, device=device)

with LineProfiler(simple_conct) as prof:
    start = time.time()
    simple_conct(X_train, batch_size, k, device)
    print(time.time() - start)
print(prof.display())

with LineProfiler(knn_batch) as prof:
    start = time.time()
    knn_batch(X_train, X_train, batch_size=batch_size, k=k, device=device)
    print(time.time() - start)
print(prof.display())