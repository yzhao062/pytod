# -*- coding: utf-8 -*-
"""Local Outlier Factor (LOF). Implemented on scikit-learn library.
"""
# Author: Yue Zhao <zhaoy@cmu.edu>
# License: BSD 2 clause


import numpy as np
import scipy as sp
import torch

from .base import BaseDetector
from .intermediate_layers import knn_batch


class LOF(BaseDetector):
    """Wrapper of scikit-learn LOF Class with more functionalities.
    Unsupervised Outlier Detection using Local Outlier Factor (LOF).

    The anomaly score of each sample is called Local Outlier Factor.
    It measures the local deviation of density of a given sample with
    respect to its neighbors.
    It is local in that the anomaly score depends on how isolated the object
    is with respect to the surrounding neighborhood.
    More precisely, locality is given by k-nearest neighbors, whose distance
    is used to estimate the local density.
    By comparing the local density of a sample to the local densities of
    its neighbors, one can identify samples that have a substantially lower
    density than their neighbors. These are considered outliers.
    See :cite:`breunig2000lof` for details.

    Parameters
    ----------
    n_neighbors : int, optional (default=20)
        Number of neighbors to use by default for `kneighbors` queries.
        If n_neighbors is larger than the number of samples provided,
        all samples will be used.

    batch_size : integer, optional (default = None)
        Number of samples to process per batch.

    device : str, optional (default = 'cpu')
        Valid device id, e.g., 'cuda:0' or 'cpu'

    Attributes
    ----------
    decision_scores_ : numpy array of shape (n_samples,)
        The outlier scores of the training data.
        The higher, the more abnormal. Outliers tend to have higher
        scores. This value is available once the detector is
        fitted.

    threshold_ : float
        The threshold is based on ``contamination``. It is the
        ``n_samples * contamination`` most abnormal samples in
        ``decision_scores_``. The threshold is calculated for generating
        binary outlier labels.

    labels_ : int, either 0 or 1
        The binary labels of the training data. 0 stands for inliers
        and 1 for outliers/anomalies. It is generated by applying
        ``threshold_`` on ``decision_scores_``.
    """

    def __init__(self, contamination=0.1, n_neighbors=5, batch_size=None,
                 device='cuda:0'):
        super(LOF, self).__init__(contamination=contamination)
        self.n_neighbors = n_neighbors
        self.batch_size = batch_size
        self.device = device

    def fit(self, X, y=None, return_time=False):
        """Fit detector. y is ignored in unsupervised methods.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.

        y : Ignored
            Not used, present for API consistency by convention.

        return_time : boolean (default=True)
            If True, set self.gpu_time to the measured GPU time.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        # todo: add one for pytorch tensor
        # X = check_array(X)
        self._set_n_classes(y)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        # find the k nearst neighbors of all samples
        knn_dist, knn_inds = knn_batch(X, X, self.n_neighbors + 1,
                                       batch_size=self.batch_size,
                                       device=self.device)
        knn_dist, knn_inds = knn_dist[:, 1:], knn_inds[:, 1:]

        end.record()
        torch.cuda.synchronize()

        # this is the index of kNN's index
        knn_inds_flat = torch.flatten(knn_inds).long()
        knn_dist_flat = torch.flatten(knn_dist)

        # for each sample, find their kNN's *kth* neighbor's distance
        # -1 is for selecting the kth distance
        knn_kth_dist = torch.index_select(knn_dist, 0, knn_inds_flat)[:, -1]
        knn_kth_inds = torch.index_select(knn_inds, 0, knn_inds_flat)[:, -1]

        # to calculate the reachable distance, we need to compare these two distances
        raw_smaller = torch.where(knn_dist_flat < knn_kth_dist)[0]

        # let's override the place where it is not the case
        # this can save one variable
        knn_dist_flat[raw_smaller] = knn_kth_dist[raw_smaller]
        # print(knn_dist_flat[:10])

        # then we need to calculate the average reachability distance

        # this result in [n, k] shape
        ar = torch.mean(knn_dist_flat.view(-1, self.n_neighbors), dim=1)

        # harmonic mean give the exact result!
        # todo: harmonic mean can be written in PyTorch as well
        ar_nn = sp.stats.hmean(
            torch.index_select(ar, 0, knn_inds_flat).view(-1,
                                                          self.n_neighbors).numpy(),
            axis=1)
        assert (len(ar_nn) == len(ar))

        self.decision_scores_ = (ar / ar_nn).cpu().numpy()

        self._process_decision_scores()

        # return GPU time in seconds
        if return_time:
            self.gpu_time = start.elapsed_time(end) / 1000
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
