"""Copula Based Outlier Detector (COPOD)
"""
# Author: Zheng Li <jk_zhengli@hotmail.com>
# Author: Yue Zhao <zhaoy@cmu.edu>
# License: BSD 2 clause

import torch

import numpy as np

from .base import BaseDetector
from .intermediate_layers import knn_full, knn_full_cpu, knn_batch


class KNN(BaseDetector):

    def __init__(self, contamination=0.1, n_neighbors=5, batch_size=None,
                 device='cuda:0'):
        super(KNN, self).__init__(contamination=contamination)
        self.n_neighbors = n_neighbors
        self.batch_size = batch_size
        self.device = device

    def fit(self, X, y=None):
        """Fit detector. y is ignored in unsupervised methods.
        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.
        y : Ignored
            Not used, present for API consistency by convention.
        Returns
        -------
        self : object
            Fitted estimator.
        """
        # todo: add one for pytorch tensor
        # X = check_array(X)
        self._set_n_classes(y)

        if self.device == 'cpu':
            # if self.batch_size is None:
            knn_dist, _ = knn_full_cpu(X, X, self.n_neighbors + 1)
            # else:
            #     knn_dist, _ = knn_batch(X, X, self.n_neighbors + 1,
            #                             batch_size=self.batch_size)

        else:
            if self.batch_size is None:
                knn_dist, _ = knn_full(X, X, self.n_neighbors + 1)
            else:
                knn_dist, _ = knn_batch(X, X, self.n_neighbors + 1,
                                        batch_size=self.batch_size)

        self.decision_scores_ = knn_dist[:, -1].cpu().numpy()
        self._process_decision_scores()
        return self

    def decision_function(self, X):
        """Predict raw anomaly score of X using the fitted detector.
         For consistency, outliers are assigned with larger anomaly scores.
        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The training input samples. Sparse matrices are accepted only
            if they are supported by the base estimator.
        Returns
        -------
        anomaly_scores : numpy array of shape (n_samples,)
            The anomaly score of the input samples.
        """
        # use multi-thread execution
        if hasattr(self, 'X_train'):
            original_size = X.shape[0]
            X = np.concatenate((self.X_train, X), axis=0)

        # return decision_scores_.ravel()
