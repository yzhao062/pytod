# -*- coding: utf-8 -*-
"""Example of using PyTOD on real-world datasets
"""
# Author: Yue Zhao <zhaoy@cmu.edu>
# License: BSD 2 clause

import warnings

warnings.filterwarnings("ignore")

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
from scipy.io import loadmat

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
from pytod.utils.data import get_roc, get_prn

# please select multiple data
# please select multiple data
file_name_list = [
    '8_celeba'
    # '9_census',
    # '11_donors',
    # '13_fraud',
]

# load PyOD models
models = {
    # 'LOF': PyOD_LOF(n_neighbors=20),
    # 'ABOD': PyOD_ABOD(n_neighbors=20),
    'HBOS': PyOD_HBOS(n_bins=50),
    # 'KNN': PyOD_KNN(n_neighbors=5),
    'PCA': PyOD_PCA(n_components=2)
}

for j in range(len(file_name_list)):
    file_name = file_name_list[j]
    # loading and vectorization
    file_path = os.path.join("datasets", "adbench", file_name + '.npz')

    data = np.load(file_path, allow_pickle=True)
    X, y = data['X'].astype('float64'), data['y'].astype(int).ravel()
    print(X.shape)

    X_torch = torch.from_numpy(X).double()

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

        print(file_name, key, roc, prn, dur)
        text_file.write(
            file_name + '|' + key + '|' + 'PyOD' + '|' + str(roc) + '|' + str(
                prn) + '|' + str(dur) + '\n')
    text_file.close()

    # get results from PyTOD
    # try to access the GPU, fall back to cpu if no gpu is available
    device = validate_device(0)
    batch_size = 30000

    # text_file = open("results.txt", "a")
    # key = 'LOF'
    # start = time.time()
    # clf = LOF(n_neighbors=20, batch_size=batch_size, device=device)
    # clf.fit(X_torch)
    # decision_scores = clf.decision_scores_
    # decision_scores = np.nan_to_num(decision_scores)
    # end = time.time()
    #
    # dur = np.round(end - start, decimals=4)
    # roc = get_roc(y, decision_scores)
    # prn = get_prn(y, decision_scores)
    #
    # print(mat_file, key, roc, prn, dur)
    # text_file.write(
    #     mat_file + '|' + key + '|' + 'PyTOD' + '|' + str(roc) + '|' + str(
    #         prn) + '|' + str(dur) + '\n')
    # text_file.close()
    # ###########################################################################
    # text_file = open("results.txt", "a")
    # key = 'ABOD'
    # start = time.time()
    # clf = ABOD(n_neighbors=20, batch_size=batch_size, device=device)
    # clf.fit(X_torch)
    # decision_scores = clf.decision_scores_
    # decision_scores = np.nan_to_num(decision_scores)
    # end = time.time()
    #
    # dur = np.round(end - start, decimals=4)
    # roc = get_roc(y, decision_scores)
    # prn = get_prn(y, decision_scores)
    #
    # print(mat_file, key, roc, prn, dur)
    # text_file.write(
    #     mat_file + '|' + key + '|' + 'PyTOD' + '|' + str(roc) + '|' + str(
    #         prn) + '|' + str(dur) + '\n')
    # text_file.close()
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

    print(file_name, key, roc, prn, dur)
    text_file.write(
        file_name + '|' + key + '|' + 'PyTOD' + '|' + str(roc) + '|' + str(
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

    print(file_name, key, roc, prn, dur)
    text_file.write(
        file_name + '|' + key + '|' + 'PyTOD' + '|' + str(roc) + '|' + str(
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

    print(file_name, key, roc, prn, dur)
    text_file.write(
        file_name + '|' + key + '|' + 'PyTOD' + '|' + str(roc) + '|' + str(
            prn) + '|' + str(dur) + '\n')
    text_file.close()

    print("The results are stored in results.txt.")
