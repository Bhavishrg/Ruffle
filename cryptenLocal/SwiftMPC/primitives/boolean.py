#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from base64 import encode
from math import floor
import sys

from numpy import dtype, int64, number, var
sys.path.insert(0, '../../')

import cryptenLocal.communicator as comm

# dependencies:
import torch
import numpy as np
import cryptenLocal as crypten
from cryptenLocal.common.functions import regular
from cryptenLocal.common.rng import generate_random_shareFromAES
from cryptenLocal.common.tensor_types import is_float_tensor, is_int_tensor, is_tensor
from cryptenLocal.common.util import torch_stack
from cryptenLocal.config import cfg
from cryptenLocal.cryptensor import CrypTensor
from cryptenLocal.cuda import CUDALongTensor
from cryptenLocal.encoder import FixedPointEncoderSwift
from cryptenLocal.SwiftMPC import modifiedCommunicator
import ouputCommitments

SENTINEL = -1


# MPC tensor where shares additive-sharings.
class BooleanSharedTensorSwift(object):

    # constructors:
    def __init__(
        self,
        tensor=None,
        size=None,
        broadcast_size=False,
        precision=None,
        src=0,
        number_of_bits = 64,
        ring_size = int(2),
    ):
        #Get current parties rank
        rank = comm.get().rank
        self.masks = []
        self.encoder = FixedPointEncoderSwift(precision_bits=precision)
        self.sharesRt = []
        self.zminusrmask = []
        self.share = []
        self.randomShare =  []
        self.number_of_bits = number_of_bits
        self.preProcessingCount = 0
        self.onlineCount = 0
        
        #Populate masks
        self.secretSharingPreprocessing(tensor, rank)

    #Arguments are value to be sent, party 1, party 2, party 3, rank
    def jmpSend(self, value1, value2, p1, p2, p3, rank):
        returnVal = 0
        if rank == p3:
            returnVal = np.uint64(modifiedCommunicator.myRecieve(p1))
            if p2 == (p3-1)%3:
                ouputCommitments.values_to_verify[0].append(returnVal)
            else:
                ouputCommitments.values_to_verify[1].append(returnVal)
            # print("recieving", rank, returnVal, rank, p2)

        elif rank == p1:
            modifiedCommunicator.mySend(value1, p3)
            returnVal = value1
        else:
            if p3 == (p2-1)%3:
                ouputCommitments.hashes_to_send_jmpsend[0].append(value2)
            else:
                ouputCommitments.hashes_to_send_jmpsend[1].append(value2)

            # print("sending", value2, rank, p3)
            returnVal = value2

        return returnVal

    def decodeShares(self):
        if comm.get().rank == 0:
            print(self.share)

    def populateValue(self, tensor, precision = None):

        #Get rank
        rank = comm.get().rank
        self.share = [np.uint64(0) for i in range(0, len(tensor))]
        for i in range(0, len(tensor)):
        
            sharesArray = self.masks[0][i]
       
            te = tensor[i]
            val = 0
            p2 = 1
            prev = sharesArray[0]
            next = sharesArray[1] 
            third = sharesArray[2]

           
            val = sharesArray[0]^sharesArray[1]^sharesArray[2]^np.uint64(tensor[i])
            if rank == 0:
                #Need to send mV to second party
                modifiedCommunicator.mySend(val, rank + 1)
            elif rank == 1:
                #Need to recieve mV from party 1
                val = modifiedCommunicator.myRecieve(rank-1)

            val = self.jmpSend(val, val, 0, 1, 2, rank)
      
            self.share[i] = np.uint64(val)
        self.onlineCount = self.onlineCount + 1

    def privateAddition(self, a, b, iteration):
        if iteration == 0:
            self.masks.append([])
            self.randomShare.append([])
            self.zminusrmask.append([])

            for i in range(0, len(a.masks[a.preProcessingCount - 1])):
                self.masks[self.preProcessingCount].append([(a.masks[a.preProcessingCount - 1][i][0]^b.masks[b.preProcessingCount - 1][i][0]), (a.masks[a.preProcessingCount - 1][i][1]^b.masks[b.preProcessingCount - 1][i][1])])
            self.preProcessingCount = self.preProcessingCount + 1
        else:
            self.share = [np.uint64(0) for i in range(0, len(a.share))]
            for i in range(0, len(self.share)):
                self.share[i] = (a.share[i]^b.share[i])
            
            self.onlineCount = self.onlineCount + 1


    def publicAddition(self, a, iteration):
        #Adds result of a and b in self
        if iteration == 1:
            self.share = [np.uint64(0) for i in range(0, len(self))]

            for i in range(0, len(self.share)):
                self.share[i] = (self.share[i]^a)
            self.onlineCount = self.onlineCount + 1

    def publicMultiplication(self, a, mult, iteration):
        #Adds result of self and a in self
        mult = np.uint64(mult)
        if iteration == 0:
            for i in range(0, len(self.masks)):         
                self.masks[self.preProcessingCount].append([np.uint64(a.masks[a.preProcessingCount - 1][i][0]&mult), np.uint64(a.masks[a.preProcessingCount - 1][i][1]&mult)])

            self.preProcessingCount = self.preProcessingCount + 1
        else:
            self.share = [np.uint64(0) for i in range(0, len(self))]
            for i in range(0, len(self.share)):
                self.share[i] = np.uint64(a.share[i]&mult)

            self.onlineCount = self.onlineCount + 1

    #TODO:Replace using ZKP
    def idealPreMult(self, rank, index, a, b):

        share1 = np.uint64(0)
        share2 = np.uint64(0)
        n = a.preProcessingCount - 1
        m = b.preProcessingCount - 1
        #Get first share
        share1 = ((a.masks[n][index][0]&b.masks[m][index][0]) ^ (a.masks[n][index][0]&b.masks[m][index][1]) ^ (b.masks[m][index][0] & a.masks[n][index][1]))


        if rank == 0:
           modifiedCommunicator.mySend(share1, rank + 1)
           share2 = modifiedCommunicator.myRecieve(rank + 2)
        elif rank == 1:
           modifiedCommunicator.mySend(share1, rank + 1)
           share2 = modifiedCommunicator.myRecieve(rank - 1)
        else:
           share2 = modifiedCommunicator.myRecieve(rank - 1)
           modifiedCommunicator.mySend(share1, rank - 2)

        (share1, share2) = (share2, share1)

        return [share1, share2]

    def computeRandomShares(self, rank, share1, share2):
        #Need to sample random vals
        ra = np.uint64(generate_random_shareFromAES("prev", rank))
        rb = np.uint64(generate_random_shareFromAES("next", rank)) 

        #Get required shares
        share1 = (share1^ra)
        share2 = (share2^rb)
        self.randomShare[self.preProcessingCount].append([share1, share2])

        return [ra, rb]     

    def getJointSharingShares(self, rank, index, randomVal1, randomVal2):
      
        share1 = 0, 
        share2 = 0, 
        share3 = 0
       
        #Here both rank 0 and rank 1 know all three shares
        if rank == 0:
            share1 = np.uint64(0)
            share2 = np.uint64(generate_random_shareFromAES("next", rank))
            share3 = np.uint64(generate_random_shareFromAES("global", rank))
            self.zminusrmask[self.preProcessingCount].append((share1^share2^share3))
           
          
        elif rank == 1:
            share1 = np.uint64(generate_random_shareFromAES("prev", rank)) 
            share2 = np.uint64(generate_random_shareFromAES("global", rank)) 
            share3 = np.uint64(0)
            self.zminusrmask[self.preProcessingCount].append((share1^share2^share3))
     
       
        else:
            share1 = np.uint64(generate_random_shareFromAES("global", rank)) 
            share2 = np.uint64(0)
            share3 = np.uint64(0)
            self.zminusrmask[self.preProcessingCount].append(np.uint64(0))


        self.zminusrmask[self.preProcessingCount][index] = self.jmpSend(self.zminusrmask[self.preProcessingCount][index], self.zminusrmask[self.preProcessingCount][index], 0, 1, 2, rank)

        self.masks[self.preProcessingCount].append([(share1^randomVal1), (share2^randomVal2), np.uint64(0)])

    def mainMultiplicationProtocol(self, rank, index, a, b):
        #Compute ys and jmp share
        n = a.onlineCount - 1
        m = b.onlineCount - 1

        y1 = ((a.masks[n][index][0]&b.share[index]) ^ (b.masks[m][index][0]&a.share[index]) ^ (self.randomShare[self.onlineCount][index][0]))
        y2 = ((a.masks[n][index][1]&b.share[index]) ^ (b.masks[m][index][1]&a.share[index]) ^ (self.randomShare[self.onlineCount][index][1]))

        ya = self.jmpSend(y2, y1, 1, 2, 0, rank)
        yb = self.jmpSend(y1, y2, 0, 2, 1, rank)
       
        nety = 0
        if rank == 0:
        
            y3 = ya
            nety = (y1^ y2 ^y3^(a.share[index]&b.share[index]))
         
        
            self.tempShares[index] = (nety^self.zminusrmask[self.onlineCount][index])

        elif rank == 1:
    
            y3 = yb
            nety = (y1^ y2 ^y3^(a.share[index]&b.share[index]))
        

            self.tempShares[index] = (nety^self.zminusrmask[self.onlineCount][index])


        #Send nety to party 2
        nety = self.jmpSend(nety, nety, 0, 1, 2, rank)
        
        self.tempShares[index] = (nety^self.zminusrmask[self.onlineCount][index])

    def privateMultiplication(self, a, b, iteration):
        rank = comm.get().rank
       
        if iteration == 0:
            self.masks.append([])
            self.randomShare.append([])
            self.zminusrmask.append([])

            #Need to get gamma shares and r shares stored. Apart from that, joint sharing shares also needed
            for i in range(0, len(a.masks[a.preProcessingCount - 1])):
                res = self.idealPreMult(rank, i, a, b)
                share1 = res[0]
                share2 = res[1]
                res1 = self.computeRandomShares(rank, share1, share2)
                self.getJointSharingShares(rank, i, res1[0], res1[1])

            self.preProcessingCount = self.preProcessingCount + 1
        else:
            self.tempShares = [np.uint64(0) for i in range(0, len(a.share))]
            for i in range(0, len(a.masks[a.onlineCount - 1])):
                self.mainMultiplicationProtocol(rank, i, a, b)
            self.share = [self.tempShares[i] for i in range(0, len(self.tempShares))]
            self.onlineCount = self.onlineCount + 1

    #Function will generate 2 shares per element in the tensor one using the previous key and the other using the next key 
    def getRandomShareForTensor(self, rank, value, index = 0):
        sharesArray = []
        val1 = 0
        val2 = 0
        cur = 1

        for i in range(0, self.number_of_bits):
            #Get shares for all parties
            sharePrev = generate_random_shareFromAES("prev", rank)%2
            shareNext = generate_random_shareFromAES("next", rank)%2

            shareThird = 0

            val1 = val1 + cur*shareNext
            val2 = val2 + cur*sharePrev
            cur = cur*2

            #Update counter of all parties
            if rank == 0:
                crypten.party1 = crypten.party1 + 1
            elif rank == 1: 
                crypten.party2 = crypten.party2 + 1
            else:
                crypten.party3 = crypten.party3 + 1

        shareThird = self.jmpSend(val1, val2, 1, 2, 0, rank)

        return [np.uint64(val2), np.uint64(val1), np.uint64(shareThird)]

        

    #Functions for preprocessing
    def secretSharingPreprocessing(self, tensor, rank):
        i = 0
        self.masks.append([])
        self.randomShare.append([])
        self.zminusrmask.append([])
        for val in tensor:
            sharesArray = self.getRandomShareForTensor(rank, tensor, i)

            self.masks[self.preProcessingCount].append(sharesArray)
            
            i = i+1
        self.preProcessingCount = self.preProcessingCount + 1
