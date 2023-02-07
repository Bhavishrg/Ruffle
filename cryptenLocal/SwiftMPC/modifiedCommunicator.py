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

totalBitsSent = 0
totalBitsRecieved = 0
numberBitsSent0 = 0
numberBitsRecieved0 = 0
numberBitsSent1 = 0
numberBitsRecieved1 = 0
numberBitsSent2 = 0
numberBitsRecieved2 = 0


def mySend(value, dst):
        if value >= 2**63:
            value = value - 2**64

        global totalBitsSent, numberBitsSent0, numberBitsSent1, numberBitsSent2
        totalBitsSent = totalBitsSent + 64
        if dst == 0:
                numberBitsSent0 = numberBitsSent0 + 64
        if dst == 1:
                numberBitsSent1 = numberBitsSent1 + 64
        if dst == 2:
                numberBitsSent2 = numberBitsSent2 + 64

        shareNext = torch.tensor(value, dtype=torch.int64)
        send_req = comm.get().isend(shareNext.contiguous(), dst=dst)
        send_req.wait()

def myRecieve(src):
        shareRecieved = torch.tensor([0], dtype=torch.int64)
        recv_req = comm.get().irecv(shareRecieved, src=src, type = 2)
        recv_req.wait()

        result = 0
        if shareRecieved < 0:
            result = np.uint64(shareRecieved.item() + 2**64)
        else:
            result = np.uint64(shareRecieved.item())
        
        global totalBitsRecieved, numberBitsRecieved0, numberBitsRecieved1, numberBitsRecieved2
        totalBitsRecieved = totalBitsRecieved + result.size*64

        if src == 0:
                numberBitsRecieved0 = numberBitsRecieved0 + 64
        if src == 1:
                numberBitsRecieved1 = numberBitsRecieved1 + 64
        if src == 2:
                numberBitsRecieved2 = numberBitsRecieved2 + 64
        
        return result

def mySend1(value, dst):
        value = np.array(value, dtype="uint64")
        value = value.astype('int64')

        global totalBitsSent, numberBitsSent0, numberBitsSent1, numberBitsSent2
        totalBitsSent = totalBitsSent + value.size*64
        if dst == 0:
                numberBitsSent0 = numberBitsSent0 + value.size*64
        if dst == 1:
                numberBitsSent1 = numberBitsSent1 + value.size*64
        if dst == 2:
                numberBitsSent2 = numberBitsSent2 + value.size*64

        shareNext = torch.tensor(value, dtype=torch.int64)
        send_req = comm.get().isend(shareNext.contiguous(), dst=dst)
        send_req.wait()

def myRecieve1(src, dim):
        shareRecieved = torch.zeros(dim, dtype=torch.int64)
        recv_req = comm.get().irecv(shareRecieved, src=src, type = 2)
        recv_req.wait()
        result = np.array(shareRecieved).astype('uint64')
        
        global totalBitsRecieved, numberBitsRecieved0, numberBitsRecieved1, numberBitsRecieved2
        totalBitsRecieved = totalBitsRecieved + result.size*64

        if src == 0:
                numberBitsRecieved0 = numberBitsRecieved0 + result.size*64
        if src == 1:
                numberBitsRecieved1 = numberBitsRecieved1 + result.size*64
        if src == 2:
                numberBitsRecieved2 = numberBitsRecieved2 + result.size*64
        
        return result