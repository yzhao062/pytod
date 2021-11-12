# -*- coding: utf-8 -*-
"""A set of utility functions to support outlier detection.
"""
# Author: Yue Zhao <zhaoy@cmu.edu>
# License: BSD 2 clause

import torch
import sklearn
from sklearn.metrics import precision_score
from sklearn.preprocessing import StandardScaler

import numpy as np
from numpy import percentile
import numbers

from sklearn.utils import column_or_1d
from sklearn.utils import check_array
from sklearn.utils import check_consistent_length


def Standardizer(X_train, mean=None, std=None, return_mean_std=False):
    if mean is None:
        mean = torch.mean(X_train, axis=0)
        std = torch.std(X_train, axis=0)
        # print(mean.shape, std.shape)
        assert (mean.shape[0] == X_train.shape[1])
        assert (std.shape[0] == X_train.shape[1])

    X_train_norm = (X_train - mean) / std
    assert (X_train_norm.shape == X_train.shape)

    if return_mean_std:
        return X_train_norm, mean, std
    else:
        return X_train_norm


def get_batch_index(n_samples, batch_size):
    """Turning 1-dimensional space into equal chunk and return the index pairs.

    Parameters
    ----------
    n_samples
    batch_size

    Returns
    -------

    """

    if n_samples <= batch_size:
        return [(0, n_samples)]

    index_tracker = []
    n_batches = int(np.ceil(n_samples // batch_size))
    print('n_batches', n_batches)
    tracker = 0
    left_index, right_index = 0, 0
    for i in range(n_batches):
        left_index = tracker * batch_size
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

def get_label_n(y, y_pred, n=None):
    """Function to turn raw outlier scores into binary labels by assign 1
    to top n outlier scores.

    Parameters
    ----------
    y : list or numpy array of shape (n_samples,)
        The ground truth. Binary (0: inliers, 1: outliers).

    y_pred : list or numpy array of shape (n_samples,)
        The raw outlier scores as returned by a fitted model.

    n : int, optional (default=None)
        The number of outliers. if not defined, infer using ground truth.

    Returns
    -------
    labels : numpy array of shape (n_samples,)
        binary labels 0: normal points and 1: outliers

    Examples
    --------
    >>> from pytod.utils.utility import get_label_n
    >>> y = [0, 1, 1, 0, 0]
    >>> y_pred = [0.1, 0.5, 0.3, 0.2, 0.7]
    >>> get_label_n(y, y_pred)
    array([0, 1, 0, 0, 1])

    """

    # enforce formats of inputs
    y = column_or_1d(y)
    y_pred = column_or_1d(y_pred)

    check_consistent_length(y, y_pred)
    y_len = len(y)  # the length of targets

    # calculate the percentage of outliers
    if n is not None:
        outliers_fraction = n / y_len
    else:
        outliers_fraction = np.count_nonzero(y) / y_len

    threshold = percentile(y_pred, 100 * (1 - outliers_fraction))
    y_pred = (y_pred > threshold).astype('int')

    return y_pred


def precision_n_scores(y, y_pred, n=None):
    """Utility function to calculate precision @ rank n.

    Parameters
    ----------
    y : list or numpy array of shape (n_samples,)
        The ground truth. Binary (0: inliers, 1: outliers).

    y_pred : list or numpy array of shape (n_samples,)
        The raw outlier scores as returned by a fitted model.

    n : int, optional (default=None)
        The number of outliers. if not defined, infer using ground truth.

    Returns
    -------
    precision_at_rank_n : float
        Precision at rank n score.

    """

    # turn raw prediction decision scores into binary labels
    y_pred = get_label_n(y, y_pred, n)

    # enforce formats of y and labels_
    y = column_or_1d(y)
    y_pred = column_or_1d(y_pred)

    return precision_score(y, y_pred)
