#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from base64 import encode
from math import floor
import sys

from numpy import dtype, int64, var
sys.path.insert(0, '../../')

import cryptenLocal.communicator as comm

# dependencies:
import torch
import numpy as np
import cryptenLocal as crypten
from cryptenLocal.common.functions import regular
from  cryptenLocal.common.rng import generate_random_shareFromAES
from cryptenLocal.common.tensor_types import is_float_tensor, is_int_tensor, is_tensor
from cryptenLocal.common.util import torch_stack
from cryptenLocal.config import cfg
from cryptenLocal.cryptensor import CrypTensor
from cryptenLocal.cuda import CUDALongTensor
from cryptenLocal.encoder import FixedPointEncoderSwift
from cryptenLocal.SwiftMPC import modifiedCommunicator
import ouputCommitments


# MPC tensor where shares additive-sharings.
#DAG List 2d array with (u, v, isVertex, data, visited)
class BFS(object):

    # constructors:
    def __init__(
        self,
        rank,
        dagList
    ):
        #Get current parties rank
        self.rank = comm.get().rank
        self.dagList = dagList
        self.M = len(dagList)

    
    def Scatter(self):
        val = 0
        for i in range(0, self.M):
            #If vertex get isVertex else store isVertex
            val = self.dagList[i][2]*self.dagList[i][3] + (1 - self.dagList[i][2])*val
            self.dagList[i][3] = val
        
    def Gather(self):
        for i in range(0, self.M):
            #If vertex get isVertex else store isVertex
            val = self.dagList[i][2]*val + (1 - self.dagList[i][2])*self.dagList[i][3]
            self.dagList[i][4] = self.dagList[i][4] + val
