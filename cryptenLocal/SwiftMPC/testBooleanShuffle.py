import sys

from pandas import array



sys.path.insert(0, '../../')

from ctypes import addressof
import cryptenLocal as crypten
import cryptenLocal.SwiftMPC as swift
import cryptenLocal.communicator as comm
import cryptenLocal.SwiftMPC.primitives.arithmetic as arithmetic
import cryptenLocal.SwiftMPC.primitives.boolean as boolean
import torch
import sys
import os
import hashlib
import time
import ouputCommitments
import shuffle


received_hashes = []

    
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
    a = [7, 3, 25, 1]
    b = [0]
    
    #Preprocessing
    a_enc = boolean.BooleanSharedTensorSwift(a, size = len(a))


    # Send around commitments of final masks
    ouputCommitments.circulateCommitmentsBoolean(rank, a_enc)

    #Populate values
    a_enc.populateValue(a)

   
    #Perform operations
    a_enc = shuffle.shufflePreProcessing(rank, a_enc)
    a_enc.masks[a_enc.onlineCount - 1] = shuffle.shuffle(rank, a_enc.masks[a_enc.onlineCount - 1])
    # #Reconstruction
    # #Open commitments and verify correctness
    ouputCommitments.verifyJmpSend(rank)
    ouputCommitments.openCommitmentsBoolean(rank, a_enc)


     
testSWIFT()

