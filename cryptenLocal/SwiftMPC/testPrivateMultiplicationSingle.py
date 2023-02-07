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
    res1 = None
    #Populate all masks
    a = [21.5, 10, 9]

    b = [5, 8, 7]

    c = [13,2,10]
 
    #Preprocessing
    a_enc = arithmetic.ArithmeticSharedTensorSwift(a, size = len(a))
    b_enc = arithmetic.ArithmeticSharedTensorSwift(b, size = len(b))
    res1 = arithmetic.ArithmeticSharedTensorSwift(c, size = len(c))
    res1.privateMultiplicationTruncation(b_enc, a_enc, 0)

    # Send around commitments of final masks
    ouputCommitments.circulateCommitments(rank, res1)

    #Populate values
    a_enc.populateValue(a)
    b_enc.populateValue(b)
    res1.populateValue(b)
    
    #Perform operations
    res1.privateMultiplicationTruncation(b_enc, a_enc, 1)

    #Verify jmp sends and open commitments
    ouputCommitments.verifyJmpSend(rank)
   
    output = ouputCommitments.openCommitments(rank, res1)



     
testSWIFT()

