from re import S
import numpy
import sys


sys.path.insert(0, '../../')

from ctypes import addressof
import cryptenLocal as crypten
import cryptenLocal.SwiftMPC as swift
import cryptenLocal.communicator as comm
import cryptenLocal.SwiftMPC.primitives.arithmetic as arithmetic
import torch
import sys
import os
import hashlib
import time
import warnings
import numpy as np

#To ignore overflow warnings
warnings.filterwarnings("ignore")

z = 2**64 - 1
z_as_int64 = numpy.uint64(z)
print(z_as_int64)

a = [[1, 2], [2, 4], [4, 5]]

b = [a[i][0] for i in range(0,len(a))]
print(b)

