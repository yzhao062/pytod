# -*- coding: utf-8 -*-
import torch
from pyod.models.knn import KNN as KNN_PyOD
from pyod.utils.data import generate_data
from pyod.utils.data import evaluate_print

import os
import sys
import time

# temporary solution for relative imports in case pyod is not installed
# if pyod is installed, no need to use the following line
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname("__file__"), '..')))

from pytod.models.knn import KNN

contamination = 0.1  # percentage of outliers
n_train = 30000  # number of training points
n_test = 5000  # number of testing points
n_features = 20
k = 10

# Generate sample data
X_train, y_train, X_test, y_test = \
    generate_data(n_train=n_train,
                  n_test=n_test,
                  n_features=n_features,
                  contamination=contamination,
                  random_state=42)

clf_name = 'KNN-PyOD'
clf = KNN_PyOD(n_neighbors=k)
start = time.time()
clf.fit(X_train)
end = time.time()
# get the prediction labels and outlier scores of the training data
y_train_pred = clf.labels_  # binary labels (0: inliers, 1: outliers)
y_train_scores = clf.decision_scores_  # raw outlier scores

# evaluate and print the results
print("\nOn Training Data:")
evaluate_print(clf_name, y_train, y_train_scores)
print('Execution time', end-start)



X_train, y_train, X_test, y_test = torch.from_numpy(X_train), \
                                   torch.from_numpy(y_train), \
                                   torch.from_numpy(X_test), \
                                   torch.from_numpy(y_test)


print()
print()
clf_name = 'KNN-PyTOD'
clf = KNN(n_neighbors=k, batch_size=10000)
# if GPU is not available, try the CPU version
# clf = KNN(n_neighbors=k, batch_size=10000, device='cpu')
start = time.time()
clf.fit(X_train)
end = time.time()
# get the prediction labels and outlier scores of the training data
y_train_pred = clf.labels_  # binary labels (0: inliers, 1: outliers)
y_train_scores = clf.decision_scores_  # raw outlier scores

# evaluate and print the results
print("\nOn Training Data:")
evaluate_print(clf_name, y_train, y_train_scores)
print('Execution time', end-start)
