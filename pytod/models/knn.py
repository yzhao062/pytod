


def knn_torch(X_train_torch, k, batch_size):
    # find the k nearest neighbors of all samples
    knn_dist, _ = knn_batch(X_train_torch, X_train_torch, k + 1,
                            batch_size=batch_size)
    # knn_dist, _ = knn_full(X_train_torch, X_train_torch, k+1,)
    return knn_dist[:, -1].numpy()


"""Copula Based Outlier Detector (COPOD)
"""
# Author: Zheng Li <jk_zhengli@hotmail.com>
# Author: Yue Zhao <zhaoy@cmu.edu>
# License: BSD 2 clause

from __future__ import division
from __future__ import print_function

import warnings
import torch
from intermediate_layers import knn_full, knn_batch
import numpy as np

from sklearn.utils import check_array

from .base import BaseDetector


class KNN(BaseDetector):

    def __init__(self, contamination=0.1, n_neighbors=5, batch_size=10000, device=):
        super(KNN, self).__init__(contamination=contamination)

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
        #todo: add one for pytorch tensor
        # X = check_array(X)
        self._set_n_classes(y)
        self.decision_scores_ = self.decision_function(X)
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

        # get the pseudo observations
        X_pseudo_obs = pseudo_obs(X)

        emp_cop = EmpiricalCopula(X_pseudo_obs, smoothing="beta")

        escores = emp_cop.cdf(X_pseudo_obs) * -1
        # self.U_r = 1 - self.U_l

        if hasattr(self, 'X_train'):
            # decision_scores_ = np.max(np.asarray([self.U_l, self.U_r]),
            #                           axis=0)[-original_size:]
            decision_scores_ = escores[-original_size:]
        else:
            # decision_scores_ = np.max(np.asarray([self.U_l, self.U_r]), axis=0)
            decision_scores_ = escores

        # self.U_l = -1 * np.log(np.apply_along_axis(ecdf, 0, X))
        # self.U_r = -1 * np.log(np.apply_along_axis(ecdf, 0, -X))
        #
        # skewness = np.sign(skew(X, axis=0))
        # self.U_skew = self.U_l * -1 * np.sign(
        #     skewness - 1) + self.U_r * np.sign(skewness + 1)
        # self.O = np.maximum(self.U_skew, np.add(self.U_l, self.U_r) / 2)
        # if hasattr(self, 'X_train'):
        #     decision_scores_ = self.O.sum(axis=1)[-original_size:]
        # else:
        #     decision_scores_ = self.O.sum(axis=1)
        return decision_scores_.ravel()
