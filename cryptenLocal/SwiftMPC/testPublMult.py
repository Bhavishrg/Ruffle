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
import ouputCommitments

#To ignore overflow warnings
warnings.filterwarnings("ignore")
    
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

    b = [5, 8, 7]

    c = [13,2,10]


    
    #Preprocessing
   
    res1 = arithmetic.ArithmeticSharedTensorSwift(c, size = len(c))
    res1.publicMultiplication(res1, 5, 0)

    # Send around commitments of final masks
    ouputCommitments.circulateCommitments(rank, res1)

    #Populate values
    res1.populateValue(c)   
    
    #Perform operations
    res1.publicMultiplication(res1, 5, 1)

 
    # Open commitments and verify correctness
    ouputCommitments.verifyJmpSend(rank)
    output = ouputCommitments.openCommitments(rank, res1)



     
testSWIFT()

