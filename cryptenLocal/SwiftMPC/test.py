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

received_hashes = []
def mySend(value, dst):
        shareNext = torch.tensor([value], dtype=torch.double)
        send_req = comm.get().isend(shareNext.contiguous(), dst=dst)
        send_req.wait()

def myRecieve(src):
        shareRecieved = torch.tensor([0], dtype=torch.double)
        recv_req = comm.get().irecv(shareRecieved, src=src, type = 2)
        recv_req.wait()
        return shareRecieved.item()


def circulateCommitments(rank, output):

    for i in range(0, len(output.masks)):
        if rank == 0:
            #Send to party 1 and 2
            hashed_string = hashlib.sha256(str(output.masks[i][0]).encode('utf-8')).hexdigest()

            smallerString = hashed_string[0:10]
            hashed_string = float(int(smallerString, 16))
            mySend(hashed_string, 1)

            hashed_string = hashlib.sha256(str(output.masks[i][1]).encode('utf-8')).hexdigest()

            smallerString = hashed_string[0:10]
            hashed_string = float(int(smallerString, 16))
            mySend(hashed_string, 2)

            #Recieve from party 1
            val = myRecieve(1)

            received_hashes.append(val)
        elif rank == 1:
            #Recieve from party 0
            val = myRecieve(0)
            #send to party 0
            hashed_string = hashlib.sha256(str(output.masks[i][1]).encode('utf-8')).hexdigest()

            smallerString = hashed_string[0:10]
            hashed_string = float(int(smallerString, 16))
            mySend(hashed_string, 0)

            received_hashes.append(val)
        else:
            #Recieve from party 0
            val = myRecieve(0)

            received_hashes.append(val)


def openCommitments(rank, output):
    ring_size = 2 ** 64
    for i in range(0, len(output.share)):
        sum = 0
        if rank == 0:
            #Send to party 1 and 2
            mySend(output.masks[i][0], 1)

            mySend(output.masks[i][1], 2)

            #Recieve from party 1
            val = myRecieve(1)
            sum = (val + output.masks[i][0] + output.masks[i][1])%ring_size

        elif rank == 1:
            #Recieve from party 0
            val = myRecieve(0)
            #send to party 0
            mySend(output.masks[i][1], 0)
            sum = (val + output.masks[i][0] + output.masks[i][1])%ring_size

        else:
            #Recieve from party 0
            val = myRecieve(0)
            sum = (val + output.masks[i][0] + output.masks[i][1])%ring_size
        output.share[i] = (output.share[i] - sum)%ring_size

    output.decodeShares()

    return output
    
@swift.run_multiprocess(world_size=3)
def testSWIFT():

    rank = comm.get().rank
    #Declare all used tensors
    a_enc = None
    b_enc = None
    c_enc = None
    res1 = None
    res2 = None
    #Populate all masks
    a = [21, 10, 9]
    a = [float(val) for val in a]

    b = [5, 8, 7]
    b = [float(val) for val in b]

    c = [13,2,10]
    c = [float(val) for val in c]


    
    #Preprocessing
    a_enc = arithmetic.ArithmeticSharedTensorSwift(a, size = len(a))
    b_enc = arithmetic.ArithmeticSharedTensorSwift(b, size = len(b))
    c_enc = arithmetic.ArithmeticSharedTensorSwift(c, size = len(c))
    res1 = arithmetic.ArithmeticSharedTensorSwift(c, size = len(c))
    res2 = arithmetic.ArithmeticSharedTensorSwift(c, size = len(c))

    res1.privateMultiplicationTruncation(b_enc, a_enc, 0)
    res2.privateMultiplicationTruncation(res1, c_enc, 0)

    # Send around commitments of final masks
    circulateCommitments(rank, res2)

    #Populate values
    a_enc.populateValue(a)
    b_enc.populateValue(b)
    c_enc.populateValue(c)
    res1.populateValue(b)
    res2.populateValue(c)

    

    #Perform operations
    res1.privateMultiplicationTruncation(b_enc, a_enc, 1)
    res2.privateMultiplicationTruncation(res1, c_enc, 1)


    # #Reconstruction
    # #Open commitments and verify correctness
    output = openCommitments(rank, res2)



     
testSWIFT()

