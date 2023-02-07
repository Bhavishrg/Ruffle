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

  
    #Populate all masks
    a = [[1, 1, 1, 0, 1], [2, 2, 1, 0, 0], [3, 3, 1, 0, 0], [4, 4, 1, 0, 0]]
    
    a_enc = []
    #Preprocessing
    for i in range(0, len(a)):
        a_enc.append([])
        a_enc[i] = boolean.BooleanSharedTensorSwift(a[i], size = len(a[i]))
    
    # Send around commitments of final masks
    ouputCommitments.circulateCommitmentsBoolean(rank, a_enc[0])

    #Populate values
    for i in range(0, len(a)):
        a_enc[i].populateValue(a[i])


   
    #Perform operations
    # a_enc = shuffle.shufflePreProcessing(rank, a_enc)
    # a_enc.masks = shuffle.shuffle(rank, a_enc.masks)
    # #Reconstruction
    # #Open commitments and verify correctness
    ouputCommitments.verifyJmpSend(rank)
    ouputCommitments.openCommitmentsBoolean(rank, a_enc[0])


     
testSWIFT()

