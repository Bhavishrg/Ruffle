import sys


sys.path.insert(0, '../../')

from ctypes import addressof
import cryptenLocal as crypten
import cryptenLocal.clarion as clarion
import cryptenLocal.communicator as comm
import cryptenLocal.SwiftMPC.primitives.arithmetic as arithmetic
import torch
import sys
import os
import hashlib
import time
import warnings
import numpy as np
import client

#To ignore overflow warnings
warnings.filterwarnings("ignore")

p = 340282366920938463463374607431768211297

def addp(a, b):
    return (a+b)%p

def multp(a, b):
    return (a*b)%p

#Get additive secret shares
def getAdditiveSecretShares(val):
    global p
    #Sample a random 128bit value
    share1 = int.from_bytes(os.urandom(16), byteorder = 'big')
    share1 = share1%p

    share2 = addp(-share1, val)

    return [share1, share2]
