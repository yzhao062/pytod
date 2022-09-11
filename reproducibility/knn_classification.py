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

n_train = 50000  # number of training points
n_test = 50000  # number of training points
n_features = 200
batch_size = 50000
k = 10
# Generate sample data
X_train = torch.randn([n_train, n_features]).float()
y_train = torch.randint(0, 2, (n_train,)).int()
X_test = torch.randn([n_test, n_features]).float()
device = validate_device(0)


def knnclf_tod(X_train, y_train, X_test, device):
    n_train, n_test = X_train.shape[0], X_test.shape[0]
    cdist_result = cdist_batch(X_test, X_train, batch_size=batch_size,
                               device=device)
    # cdist_result = cdist_batch(X_train, X_test, batch_size=batch_size, device='cpu')

    bottomk_dist, bottomk_ind = bottomk_batch(cdist_result, k,
                                              batch_size=batch_size,
                                              device=device)
    # print(cdist_result, cdist_result.shape)
    # print(bottomk_ind.shape, bottomk_dist, time.time() - start)

    y_train_repeat = y_train.repeat(1, n_test).reshape(n_test, n_train)
    # print(y_train_repeat)
    # print(y_train_repeat.shape)

    knn_results = y_train_repeat.gather(1, bottomk_ind.long())
    knn_vote = torch.sum(knn_results, dim=1) / k

    # get the pred results of kNN by TOD
    pred = (knn_vote >= 0.5).int()
    return pred


start = time.time()
pred = knnclf_tod(X_train, y_train, X_test, device)
print(time.time() - start)


from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=k)
start = time.time()
neigh.fit(X_train.numpy(), y_train.numpy())
neigh.predict(X_test.numpy())
print(time.time() - start)