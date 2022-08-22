from torch import cdist

from pytod.models.basic_operators import bottomk


def knn_full(A, B, k=5, p=2.0, device=None):
    """Get kNN in the non-batch way

    Parameters
    ----------
    A
    B
    k
    p
    device

    Returns
    -------

    """
    dist_c = cdist(A.to(device), B.to(device), p=p)
    btk_d, btk_i = bottomk(dist_c, k=k, device=device)
    return btk_d.cpu(), btk_i.cpu()


