# -*- coding: utf-8 -*-
"""Implementation of some basic operators.
"""
# Author: Yue Zhao <zhaoy@cmu.edu>
# License: BSD 2 clause


import torch
from torch import cdist as torch_cdist

# disable autograd since no grad is needed
torch.set_grad_enabled(False)


def cdist(a, b=None, p=2, device='cpu'):
    """Basic cdist without using batching

    Parameters
    ----------
    a
    b
    p

    Returns
    -------

    """
    if b is None:
        b = a
        return torch_cdist(b.to(device), b.to(device), p=p)
    else:
        return torch_cdist(a.to(device), b.to(device), p=p)


# def cdist_s(a, b):
#     """Memory saving version of cdist
#
#     Parameters
#     ----------
#     a
#     b
#
#     Returns
#     -------
#
#     """
#     norm_a = torch.norm(a, dim=1).reshape(a.shape[0], 1)
#     norm_b = torch.norm(b, dim=1).reshape(1, b.shape[0])
#
#     w = norm_a ** 2 + norm_b ** 2 - 2 * torch.matmul(a, b.T)
#     return torch.sqrt(w)


def topk(A, k, dim=1, device='cpu'):
    """Returns the k the largest elements of the given input tensor along a given dimension.

    Parameters
    ----------
    A
    k
    dim

    Returns
    -------
    values : tensor of shape (n_samples, k)
        Top k values.

    index : tensor of shape (n_samples, k)
        Top k indexes.

    """
    if len(A.shape) == 1:
        dim = 0
    tk = torch.topk(A.to(device), k, dim=dim)
    return tk[0].cpu(), tk[1].cpu()


def bottomk(A, k, dim=1, device='cpu'):
    if len(A.shape) == 1:
        dim = 0
    # tk = torch.topk(A * -1, k, dim=dim)
    # see parameter https://pytorch.org/docs/stable/generated/torch.topk.html
    tk = torch.topk(A.to(device), k, dim=dim, largest=False)
    return tk[0].cpu(), tk[1].cpu()


def bottomk_cpu(A, k, dim=1):
    if len(A.shape) == 1:
        dim = 0
    # tk = torch.topk(A * -1, k, dim=dim)
    # see parameter https://pytorch.org/docs/stable/generated/torch.topk.html
    tk = torch.topk(A, k, dim=dim, largest=False)
    return tk[0], tk[1]


def bottomk_low_prec(A, k, dim=1, mode='half', sort_value=False, device='cpu'):
    # in lower precision
    if mode == 'half':
        # do conversion first
        A_GPU = A.half().to(device)

    else:
        A_GPU = A.float().to(device)

    bottomk_dist, bottomk_indices = bottomk(A_GPU, k + 1)

    # get all the ambiguous indices with 2-element assumption
    amb_indices_p1 = torch.where(bottomk_dist[:, k] <= bottomk_dist[:, k - 1])[
        0]
    amb_indices_m1 = \
        torch.where(bottomk_dist[:, k - 2] >= bottomk_dist[:, k - 1])[0]

    # there might be repetition, so we need to find the unique element only
    amb_indices = torch.unique(torch.cat((amb_indices_p1, amb_indices_m1)))

    print("ambiguous indices", len(amb_indices))

    # recal_cdist = cdist_dist[amb_indices, :].double()
    A_GPU_recal = A[amb_indices, :].to(device)

    _, bottomk_indices[amb_indices, :k] = bottomk(A_GPU_recal, k)

    # drop the last bit k+1
    bottomk_indices = bottomk_indices[:, :k].cpu()

    # select by indices for bottom distance
    # https://stackoverflow.com/questions/58523290/select-mask-different-column-index-in-every-row
    bottomk_dist = A.gather(1, bottomk_indices)

    if sort_value:
        bottomk_dist_sorted, bottomk_indices_argsort = torch.sort(bottomk_dist,
                                                                  dim=dim)
        bottomk_indices_sorted = bottomk_indices.gather(1,
                                                        bottomk_indices_argsort)
        return bottomk_dist_sorted, bottomk_indices_sorted
    else:
        return bottomk_dist, bottomk_indices


def topk_low_prec(A, k, dim=1, mode='half', sort_value=False, device='cpu'):
    # in lower precision
    if mode == 'half':
        # do conversion first
        A_GPU = A.half().to(device)

    else:
        A_GPU = A.float().to(device)

    print(A_GPU)

    topk_dist, topk_indices = topk(A_GPU, k + 1)

    # topk(A, k+1)
    # print(A)
    # get all the ambiguous indices with 2-element assumption
    amb_indices_p1 = torch.where(topk_dist[:, k] >= topk_dist[:, k - 1])[0]
    amb_indices_m1 = torch.where(topk_dist[:, k - 2] <= topk_dist[:, k - 1])[0]

    # there might be repetition, so we need to find the unique element only
    amb_indices = torch.unique(torch.cat((amb_indices_p1, amb_indices_m1)))

    print("ambiguous indices", len(amb_indices))

    A_GPU_recal = A[amb_indices, :].to(device)
    # recal_cdist = cdist_dist[amb_indices, :].double()
    _, topk_indices[amb_indices, :k] = topk(A_GPU_recal, k)

    # drop the last bit k+1
    topk_indices = topk_indices[:, :k].cpu()

    # select by indices for bottom distance
    # https://stackoverflow.com/questions/58523290/select-mask-different-column-index-in-every-row

    topk_dist = A.gather(1, topk_indices)

    if sort_value:
        topk_dist_sorted, topk_indices_argsort = torch.sort(topk_dist, dim=dim,
                                                            descending=True)
        topk_indices_sorted = topk_indices.gather(1, topk_indices_argsort)
        return topk_dist_sorted, topk_indices_sorted
    else:
        return topk_dist, topk_indices


def intersec1d(t1_orig, t2_orig, assume_unique=False, device='cpu'):
    t1_orig = t1_orig.to(device)
    t2_orig = t2_orig.to(device)
    # adapted from https://github.com/numpy/numpy/blob/v1.19.0/numpy/lib/arraysetops.py#L347-L441
    if assume_unique:
        aux = torch.cat((t1_orig, t2_orig))
    else:
        t1 = torch.unique(t1_orig)
        t2 = torch.unique(t2_orig)
        aux = torch.cat((t1, t2))

    aux = torch.sort(aux)[0]

    mask = aux[1:] == aux[:-1]
    int1d = aux[:-1][mask]
    # print(t1)
    # for i in int1d:
    #     print('t1', (i==t1_orig).nonzero())
    #     print('t2', (i==t2_orig).nonzero())

    return int1d.cpu()


def intersecmulti(A, B, assume_unique=False):
    assert (A.shape[0] == B.shape[0])
    n_samples = A.shape[0]

    intersec = []
    intersec_count = []
    for i in range(n_samples):
        intersec.append(intersec1d(A[i, :], B[i, :]))
        intersec_count.append(len(intersec[-1]))
    return intersec, intersec_count


def post_check_intersection1d(t1, t2, intersect):
    for i in intersect:
        if i not in t1 or i not in t2:
            assert ('intersection error')


def ecdf_multiple(X, device='cpu'):
    """Get the ECDF results per feature. GPU version is way faster than the
    CPU version.

    Parameters
    ----------
    X : numpy array of shape (n_samples, n_features)
        The input samples.

    device : string of device (default='cpu')

    Returns
    -------

    ECDF : numpy array of shape (n_samples, n_features)
    """
    argx_tensor = torch.argsort(X.to(device), dim=0)
    y_tensor = torch.linspace(1 / X.shape[0], 1, X.shape[0]).to(device)
    return y_tensor[argx_tensor].cpu()


def svd_randomized(M, k=10, device='cpu'):
    # http://gregorygundersen.com/blog/2019/01/17/randomized-svd/
    # http://algorithm-interest-group.me/assets/slides/randomized_SVD.pdf
    n_samples, n_dims = M.shape[0], M.shape[1]
    P = torch.randn([n_dims, k]).float().to(device)
    M_P = torch.mm(M, P)
    Q, _ = torch.qr(M_P)
    B = torch.mm(Q.T, M)
    U, S, V = torch.svd(B)
    U = torch.mm(Q, U)

    return U, S, V


def histt(a, bins=10, density=True, device='cpu'):
    def diff(a):
        # https://discuss.pytorch.org/t/equivalent-function-like-numpy-diff-in-pytorch/35327
        return a[1:] - a[:-1]

    # https://github.com/numpy/numpy/blob/v1.19.0/numpy/lib/histograms.py#L677-L928
    # for i in range(a.shape[1]):
    hist = torch.histc(a, bins=bins)
    # normalize histogram to sum to 1
    # hist = torch.true_divide(hist, hist.sum())
    bin_edges = torch.linspace(a.min(), a.max(), steps=bins + 1).to(device)
    if density:
        hist_sum = hist.sum()
        db = diff(bin_edges).to(device)
        return torch.true_divide(hist, db) / hist_sum, bin_edges

    else:
        return hist, bin_edges
