#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import sys
sys.path.insert(0, '../../')

import cryptenLocal as crypten
import torch
from torch.optim.optimizer import required


class Optimizer(torch.optim.Optimizer):
    r"""Base class for all optimizers.
    .. warning::
        Parameters need to be specified as collections that have a deterministic
        ordering that is consistent between runs. Examples of objects that don't
        satisfy those properties are sets and iterators over values of dictionaries.
    Arguments:
        params (iterable): an iterable of :class:`torch.Tensor` s,
            :class:`dict` s, or :class:`crypten.CrypTensor`s. Specifies what Tensors
            should be optimized.
        defaults: (dict): a dict containing default values of optimization
            options (used when a parameter group doesn't specify them).

    Note: This optimizer is adapted from torch.optim.Optimizer to work with CrypTensors
    """

    def add_param_group(self, param_group):
        r"""Add a param group to the :class:`Optimizer` s `param_groups`.
        This can be useful when fine tuning a pre-trained network as frozen layers can be made
        trainable and added to the :class:`Optimizer` as training progresses.
        Arguments:
            param_group (dict): Specifies what Tensors should be optimized along with group
            specific optimization options.
        """
        assert isinstance(param_group, dict), "param group must be a dict"

        params = param_group["params"]
        if isinstance(params, (torch.Tensor, crypten.CrypTensor)):
            param_group["params"] = [params]
        elif isinstance(params, set):
            raise TypeError(
                "optimizer parameters need to be organized in ordered collections, but "
                "the ordering of tensors in sets will change between runs. Please use a list instead."
            )
        else:
            param_group["params"] = list(params)

        for param in param_group["params"]:
            if not isinstance(param, (torch.Tensor, crypten.CrypTensor)):
                raise TypeError(
                    "optimizer can only optimize Tensors, "
                    "but one of the params is " + torch.typename(param)
                )

        for name, default in self.defaults.items():
            if default is required and name not in param_group:
                raise ValueError(
                    "parameter group didn't specify a value of required optimization parameter "
                    + name
                )
            else:
                param_group.setdefault(name, default)

        self.param_groups.append(param_group)
