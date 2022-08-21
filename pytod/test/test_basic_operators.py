# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function

import os
import sys
import unittest

import numpy as np
import torch
from numpy.testing import assert_equal
from numpy.testing import assert_raises

# temporary solution for relative imports in case pytod is not installed
# if pytod is installed, no need to use the following line
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pytod.utils.data import generate_data
from pytod.models.basic_operators import cdist
from pytod.models.basic_operators import topk
from pytod.utils.utility import validate_device


class TestCDIST(unittest.TestCase):

    def setUp(self):
        self.X = torch.Tensor([[1, 1], [2, 2], [3, 3]])
        self.device = validate_device(0)

    def test_calc(self):
        dist = cdist(self.X, self.X, p=2, device=self.device)
        assert (dist.shape[0] - dist.shape[1] == 0)
        assert (torch.diagonal(dist).sum() == 0)

class TestTOPK(unittest.TestCase):

    def setUp(self):
        self.X = torch.Tensor([[1, 1], [2, 2], [3, 3]])
        self.device = validate_device(0)
        self.dist = cdist(self.X, self.X, p=2, device=self.device)

    def test_calc(self):
        topk_val, topk_ind = topk(self.dist, k=1, device=self.device)
        # print(topk_ind)
        # print(topk_ind.cpu().numpy().tolist())
        assert (topk_ind.cpu().numpy().tolist() == [[2], [0], [0]])
        # print(topk_val)
        # print(np.round(topk_val.cpu().numpy(), decimals=4).tolist())
        assert (np.round(topk_val.cpu().numpy(), decimals=4).tolist() == [[2.828399896621704], [1.414199948310852], [2.828399896621704]])

