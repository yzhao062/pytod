import os
import sys
import time

import numpy as np
import torch
from pyod.utils.data import generate_data

# temporary solution for relative imports in case pyod is not installed
# if pyod is installed, no need to use the following line
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname("__file__"), '..')))

from pytod.models.abod import ABOD
from pytod.models.lof import LOF
from pytod.models.knn import KNN
from pytod.models.pca import PCA
from pytod.models.hbos import HBOS
from pytod.utils.utility import validate_device

# define the synthetic data here
contamination = 0.1  # percentage of outliers
n_train = 10000  # number of training points
n_features = 200
k = 20
batch_size = 30000

# Generate sample data
X, y = generate_data(n_train=n_train,
                     n_features=n_features,
                     contamination=contamination,
                     train_only=True,
                     random_state=42)

mat_file = str(n_train) + '_' + str(n_features)
X_torch = torch.from_numpy(X).float()

device = validate_device(0)

key = 'KNN'
start = time.time()
clf = KNN(n_neighbors=20, batch_size=batch_size, device=device)
clf.fit(X_torch, return_time=True)
decision_scores = clf.decision_scores_
decision_scores = np.nan_to_num(decision_scores)
end = time.time()
print('kNN total time', end - start)
print('kNN GPU time', clf.gpu_time)

key = 'HBOS'
start = time.time()
clf = HBOS(n_bins=50, alpha=0.1, device=device)
clf.fit(X_torch, return_time=True)
decision_scores = clf.decision_scores_
decision_scores = np.nan_to_num(decision_scores)
end = time.time()
print('HBOS total time', end - start)
print('HBOS GPU time', clf.gpu_time)

key = 'PCA'
start = time.time()
clf = PCA(n_components=5, device=device)
clf.fit(X_torch, return_time=True)
decision_scores = clf.decision_scores_
decision_scores = np.nan_to_num(decision_scores)
end = time.time()
print('PCA total time', end - start)
print('PCA GPU time', clf.gpu_time)

key = 'LOF'
start = time.time()
clf = LOF(n_neighbors=20, batch_size=batch_size, device=device)
clf.fit(X_torch, return_time=True)
decision_scores = clf.decision_scores_
decision_scores = np.nan_to_num(decision_scores)
end = time.time()
print('LOF total time', end - start)
print('LOF GPU time', clf.gpu_time)

key = 'ABOD'
start = time.time()
clf = ABOD(n_neighbors=20, batch_size=batch_size, device=device)
clf.fit(X_torch, return_time=True)
decision_scores = clf.decision_scores_
decision_scores = np.nan_to_num(decision_scores)
end = time.time()
print('ABOD total time', end - start)
print('ABOD GPU time', clf.gpu_time)


