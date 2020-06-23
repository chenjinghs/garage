"""Test CategoricalCNNModule."""
import pickle

import numpy as np
import pytest
import torch
from torch.distributions import Categorical
import torch.nn as nn

from garage.torch.modules.categorical_cnn_module import CategoricalCNNModule


class TestCategoricalCNNModule:
    """Test CategoricalCNNModule."""

    def setup_method(self):
        self.batch_size = 64
        self.input_width = 32
        self.input_height = 32
        self.in_channel = 3
        self.dtype = torch.float32
        self.input = torch.zeros(
            (self.batch_size, self.in_channel, self.input_height,
             self.input_width),
            dtype=self.dtype)  # minibatch size 64, image size [3, 32, 32]

    def test_dist(self):
        model = CategoricalCNNModule(
            input_var=self.input,
            # input_dim=1,
            output_dim=1,
            kernel_sizes=((3), ),
            hidden_channels=((5), ),
            strides=(1, ),
        )
        dist = model(self.input)
        assert isinstance(dist, Categorical)

    def test_is_pickleable(self):
        pass
