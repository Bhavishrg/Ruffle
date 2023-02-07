from pickle import bytes_types
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

totalBitsSent = 0
totalBitsRecieved = 0
numberBitsSent0 = 0
numberBitsRecieved0 = 0
numberBitsSent1 = 0
numberBitsRecieved1 = 0
numberBitsSent2 = 0
numberBitsRecieved2 = 0
numberBitsSent3 = 0
numberBitsRecieved3 = 0

#To ignore overflow warnings
warnings.filterwarnings("ignore")
def mySend(value, dst):
        byteString = value

        global totalBitsSent, numberBitsSent0, numberBitsSent1, numberBitsSent2, numberBitsSent3
        totalBitsSent = totalBitsSent + 128
        if dst == 0:
                numberBitsSent0 = numberBitsSent0 + 128
        if dst == 1:
                numberBitsSent1 = numberBitsSent1 + 128
        if dst == 2:
                numberBitsSent2 = numberBitsSent2 + 128
        if dst == 3:
                numberBitsSent3 = numberBitsSent3 + 128
        #Need to send 128 bit value so split into 2 parts 
        if not isinstance(value, (bytes, bytearray)):
            byteString = value.to_bytes(16, byteorder = 'big')
      
        upper = int.from_bytes(byteString[0:8], byteorder = 'big')
        lower = int.from_bytes(byteString[8:16], byteorder = 'big')

        if upper >= 2**63:
            upper = upper - 2**64
        if lower >= 2**63:
            lower = lower - 2**64
        upperVal = torch.tensor(upper, dtype=torch.int64)
        send_req = comm.get().isend(upperVal.contiguous(), dst=dst)
        send_req.wait()

        lowerVal = torch.tensor(lower, dtype=torch.int64)
        send_req = comm.get().isend(lowerVal.contiguous(), dst=dst)
        send_req.wait()

def myRecieve(src):

        global totalBitsRecieved, numberBitsRecieved0, numberBitsRecieved1, numberBitsRecieved2, numberBitsRecieved3
        
        totalBitsRecieved = totalBitsRecieved + 128

        if src == 0:
                numberBitsRecieved0 = numberBitsRecieved0 + 128
        if src == 1:
                numberBitsRecieved1 = numberBitsRecieved1 + 128
        if src == 2:
                numberBitsRecieved2 = numberBitsRecieved2 + 128
        if src == 3:
                numberBitsRecieved3 = numberBitsRecieved3 + 128

        upperVal = torch.tensor([0], dtype=torch.int64)
        upperValReq = comm.get().irecv(upperVal, src=src, type = 2)
        upperValReq.wait()

        lowerVal = torch.tensor([0], dtype=torch.int64)
        lowerValReq = comm.get().irecv(lowerVal, src=src, type = 2)
        lowerValReq.wait()

        byteString = (0).to_bytes(16, byteorder='big')

        upperValInt = upperVal.item()
        lowerValInt = lowerVal.item()
        if upperValInt < 0:
            upperValInt = upperValInt + 2**64
        if lowerValInt < 0:
            lowerValInt = lowerValInt + 2**64

        byteResult = b""
        byteString1 = upperValInt.to_bytes(8, byteorder = 'big')
        byteString2 = lowerValInt.to_bytes(8, byteorder = 'big')

        byteResult = byteString1 + byteString2

        return int.from_bytes(byteResult, byteorder = 'big')


def mySend1(value, dst):
        
        sendVal = []
        for i in range(0, len(value)):
            curArr = []
            for j in range(0, len(value[0])):
                    byteString = value[i][j].to_bytes(16, byteorder = 'big')
                    upper = int.from_bytes(byteString[0:8], byteorder = 'big')
                    lower = int.from_bytes(byteString[8:16], byteorder = 'big')
                    curArr.append([upper, lower])
            sendVal.append(curArr)
      
        value = np.array(sendVal, dtype="uint64")
        value = value.astype('int64')

        global totalBitsSent, numberBitsSent0, numberBitsSent1, numberBitsSent2, numberBitsSent3
        totalBitsSent = totalBitsSent + value.size*64
        if dst == 0:
                numberBitsSent0 = numberBitsSent0 + value.size*64
        if dst == 1:
                numberBitsSent1 = numberBitsSent1 + value.size*64
        if dst == 2:
                numberBitsSent2 = numberBitsSent2 + value.size*64
        if dst == 3:
                numberBitsSent3 = numberBitsSent3 + value.size*64


        shareNext = torch.tensor(value, dtype=torch.int64)
        send_req = comm.get().isend(shareNext.contiguous(), dst=dst)
        send_req.wait()

def myRecieve1(src, dim):
        shareRecieved = torch.zeros(dim, dtype=torch.int64)
        recv_req = comm.get().irecv(shareRecieved, src=src, type = 2)
        recv_req.wait()
        result = np.array(shareRecieved).astype('uint64')

        global totalBitsRecieved, numberBitsRecieved0, numberBitsRecieved1, numberBitsRecieved2, numberBitsRecieved3
        totalBitsRecieved = totalBitsRecieved + result.size*64

        if src == 0:
                numberBitsRecieved0 = numberBitsRecieved0 + result.size*64
        if src == 1:
                numberBitsRecieved1 = numberBitsRecieved1 + result.size*64
        if src == 2:
                numberBitsRecieved2 = numberBitsRecieved2 + result.size*64
        if src == 3:
                numberBitsRecieved3 = numberBitsRecieved3 + result.size*64
        
        resVal = []
        for i in range(0, dim[0]):
            cur = []
            for j in range(0, dim[1]):

                byteResult = b""
                byteString1 = int(result[i][j][0]).to_bytes(8, byteorder = 'big')
                byteString2 = int(result[i][j][1]).to_bytes(8, byteorder = 'big')
                byteResult = byteString1 + byteString2  

                cur.append(int.from_bytes(byteResult, byteorder = 'big'))
            resVal.append(cur)

        return resVal