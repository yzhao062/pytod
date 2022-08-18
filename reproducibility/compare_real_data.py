# -*- coding: utf-8 -*-
"""Example of using PyTOD on real-world datasets
"""
# Author: Yue Zhao <zhaoy@cmu.edu>
# License: BSD 2 clause

import os
import sys
import time

import numpy as np
import torch
from pyod.models.abod import ABOD as PyOD_ABOD
from pyod.models.hbos import HBOS as PyOD_HBOS
from pyod.models.knn import KNN as PyOD_KNN
from pyod.models.lof import LOF as PyOD_LOF
from pyod.models.pca import PCA as PyOD_PCA
from pyod.utils.utility import precision_n_scores
from scipy.io import loadmat
from sklearn.metrics import roc_auc_score

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


def get_roc(y, y_pred):
    from sklearn.utils import column_or_1d
    from sklearn.utils import check_consistent_length
    y = column_or_1d(y)
    y_pred = column_or_1d(y_pred)
    check_consistent_length(y, y_pred)

    return np.round(roc_auc_score(y, y_pred), decimals=4)


def get_prn(y, y_pred):
    from sklearn.utils import column_or_1d
    from sklearn.utils import check_consistent_length
    y = column_or_1d(y)
    y_pred = column_or_1d(y_pred)
    check_consistent_length(y, y_pred)

    return np.round(precision_n_scores(y, y_pred), decimals=4)


# please select multiple data
mat_file_list = [
    # 'annthyroid.mat',
    # 'arrhythmia.mat',
    # 'breastw.mat',
    # 'glass.mat',
    # 'ionosphere.mat',
    # 'letter.mat',
    # 'lympho.mat',
    # 'mammography.mat',
    'mnist.mat',
    # 'musk.mat',
    # 'optdigits.mat',
    # 'pendigits.mat',
    # 'pima.mat',
    # 'satellite.mat',
    # 'satimage-2.mat',
    # 'shuttle.mat',
    # 'smtp_n.mat',
    # 'speech.mat',
    # 'thyroid.mat',
    # 'vertebral.mat',
    # 'vowels.mat',
    # 'wbc.mat',
    # 'wine.mat',
]

# load PyOD models
models = {
    'LOF': PyOD_LOF(n_neighbors=20),
    'ABOD': PyOD_ABOD(n_neighbors=20),
    'HBOS': PyOD_HBOS(n_bins=50),
    'KNN': PyOD_KNN(n_neighbors=5),
    'PCA': PyOD_PCA(n_components=5)
}

for j in range(len(mat_file_list)):
    mat_file = mat_file_list[j]
    # loading and vectorization
    mat = loadmat(os.path.join("datasets", "ODDS", mat_file))

    X = mat['X'].astype('float')
    y = mat['y'].ravel()
    X_torch = torch.from_numpy(X).float()

    # initialize the output file
    text_file = open("results.txt", "a")
    text_file.write(
        'file' + '|' + 'algorithm' + '|' + 'system' + '|' + 'ROC' + '|' + 'PRN' + '|' + 'Runtime' + '\n')

    for key in models.keys():
        clf = models[key]

        start = time.time()
        clf.fit(X)
        decision_scores = clf.decision_scores_
        decision_scores = np.nan_to_num(decision_scores)
        end = time.time()

        dur = np.round(end - start, decimals=4)
        roc = get_roc(y, decision_scores)
        prn = get_prn(y, decision_scores)

        print(mat_file, key, roc, prn, dur)
        text_file.write(
            mat_file + '|' + key + '|' + 'PyOD' + '|' + str(roc) + '|' + str(
                prn) + '|' + str(dur) + '\n')
    text_file.close()

    # get results from PyTOD
    # try to access the GPU, fall back to cpu if no gpu is available
    device = validate_device(0)
    batch_size = 30000

    text_file = open("results.txt", "a")
    key = 'LOF'
    start = time.time()
    clf = LOF(n_neighbors=20, batch_size=batch_size, device=device)
    clf.fit(X_torch)
    decision_scores = clf.decision_scores_
    decision_scores = np.nan_to_num(decision_scores)
    end = time.time()

    dur = np.round(end - start, decimals=4)
    roc = get_roc(y, decision_scores)
    prn = get_prn(y, decision_scores)

    print(mat_file, key, roc, prn, dur)
    text_file.write(
        mat_file + '|' + key + '|' + 'PyTOD' + '|' + str(roc) + '|' + str(
            prn) + '|' + str(dur) + '\n')
    text_file.close()
    ###########################################################################
    text_file = open("results.txt", "a")
    key = 'ABOD'
    start = time.time()
    clf = ABOD(n_neighbors=20, batch_size=batch_size, device=device)
    clf.fit(X_torch)
    decision_scores = clf.decision_scores_
    decision_scores = np.nan_to_num(decision_scores)
    end = time.time()

    dur = np.round(end - start, decimals=4)
    roc = get_roc(y, decision_scores)
    prn = get_prn(y, decision_scores)

    print(mat_file, key, roc, prn, dur)
    text_file.write(
        mat_file + '|' + key + '|' + 'PyTOD' + '|' + str(roc) + '|' + str(
            prn) + '|' + str(dur) + '\n')
    text_file.close()
    ###########################################################################
    text_file = open("results.txt", "a")
    key = 'HBOS'
    start = time.time()
    clf = HBOS(n_bins=50, alpha=0.1, device=device)
    clf.fit(X_torch)
    decision_scores = clf.decision_scores_
    decision_scores = np.nan_to_num(decision_scores)
    end = time.time()

    dur = np.round(end - start, decimals=4)
    roc = get_roc(y, decision_scores)
    prn = get_prn(y, decision_scores)

    print(mat_file, key, roc, prn, dur)
    text_file.write(
        mat_file + '|' + key + '|' + 'PyTOD' + '|' + str(roc) + '|' + str(
            prn) + '|' + str(dur) + '\n')
    text_file.close()
    # #############################################################################################
    text_file = open("results.txt", "a")
    key = 'KNN'
    start = time.time()
    clf = KNN(n_neighbors=5, batch_size=batch_size, device=device)
    clf.fit(X_torch)
    decision_scores = clf.decision_scores_
    decision_scores = np.nan_to_num(decision_scores)
    end = time.time()

    dur = np.round(end - start, decimals=4)
    roc = get_roc(y, decision_scores)
    prn = get_prn(y, decision_scores)

    print(mat_file, key, roc, prn, dur)
    text_file.write(
        mat_file + '|' + key + '|' + 'PyTOD' + '|' + str(roc) + '|' + str(
            prn) + '|' + str(dur) + '\n')
    text_file.close()
    # #############################################################################################
    text_file = open("results.txt", "a")
    key = 'PCA'
    start = time.time()
    clf = PCA(n_components=5)
    clf.fit(X_torch)
    decision_scores = clf.decision_scores_
    decision_scores = np.nan_to_num(decision_scores)
    end = time.time()

    dur = np.round(end - start, decimals=4)
    roc = get_roc(y, decision_scores)
    prn = get_prn(y, decision_scores)

    print(mat_file, key, roc, prn, dur)
    text_file.write(
        mat_file + '|' + key + '|' + 'PyTOD' + '|' + str(roc) + '|' + str(
            prn) + '|' + str(dur) + '\n')
    text_file.close()

    print("The results are stored in results.txt.")
