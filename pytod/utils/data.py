import torch
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.utils import column_or_1d
from sklearn.utils import check_consistent_length

from pyod.utils.utility import precision_n_scores
from pyod.utils.data import evaluate_print as evaluate_print_np


def evaluate_print(clf_name, y, y_pred):
    """Utility function for evaluating and printing the results for examples.
    Default metrics include ROC and Precision @ n

    Parameters
    ----------
    clf_name : str
        The name of the detector.

    y : list or numpy array of shape (n_samples,)
        The ground truth. Binary (0: inliers, 1: outliers).

    y_pred : list or numpy array of shape (n_samples,)
        The raw outlier scores as returned by a fitted model.

    """

    if torch.is_tensor(y):
        y = y.cpu().numpy()
        y_pred = y_pred.cpu().numpy()
    return evaluate_print_np(clf_name, y, y_pred)


def get_roc(y, y_pred):
    """Utility function for evaluating the results for examples.
        Default metrics include ROC

        Parameters
        ----------
        y : list or numpy array of shape (n_samples,)
            The ground truth. Binary (0: inliers, 1: outliers).

        y_pred : list or numpy array of shape (n_samples,)
            The raw outlier scores as returned by a fitted model.

        """
    y = column_or_1d(y)
    y_pred = column_or_1d(y_pred)
    check_consistent_length(y, y_pred)

    return np.round(roc_auc_score(y, y_pred), decimals=4)


def get_prn(y, y_pred):
    """Utility function for evaluating the results for examples.
        Default metrics include P@N

        Parameters
        ----------
        y : list or numpy array of shape (n_samples,)
            The ground truth. Binary (0: inliers, 1: outliers).

        y_pred : list or numpy array of shape (n_samples,)
            The raw outlier scores as returned by a fitted model.

        """
    y = column_or_1d(y)
    y_pred = column_or_1d(y_pred)
    check_consistent_length(y, y_pred)

    return np.round(precision_n_scores(y, y_pred), decimals=4)
