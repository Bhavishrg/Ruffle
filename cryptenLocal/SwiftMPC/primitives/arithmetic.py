#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from base64 import encode
from math import floor
import sys

from numpy import dtype, int64, var
sys.path.insert(0, '../../')

import cryptenLocal.communicator as comm

# dependencies:
import torch
import numpy as np
import cryptenLocal as crypten
from cryptenLocal.common.functions import regular
from  cryptenLocal.common.rng import generate_random_shareFromAES
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
class ArithmeticSharedTensorSwift(object):

    # constructors:
    def __init__(
        self,
        tensor=None,
        size=None,
        broadcast_size=False,
        precision=None,
        src=0,
        ring_size = int(2 ** 64),
    ):
        #Get current parties rank
        rank = comm.get().rank
        self.masks = []
        self.encoder = FixedPointEncoderSwift(precision_bits=precision)
        self.sharesRt = []
        self.zminusrmask = []
        self.share = []
        self.randomShare =  []
        self.preprocessingCount = 0
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
     
        self.share = self.encoder.decode(self.share)
        if comm.get().rank == 0:
            print(self.share)

    def populateValue(self, tensor, precision = None):
        # encode the input tensor:
        tensor = self.encoder.encode(tensor)

        #Get rank
        rank = comm.get().rank
        self.share = [np.uint64(0) for i in range(0, len(tensor))]
        
        for i in range(0, len(tensor)):
        
            (sharePrev, shareNext, shareThird) = self.masks[0][i]
           
            mV = (np.uint64(tensor[i]) + sharePrev + shareNext + shareThird)
            
            if rank == 0:
                #Need to send mV to second party
                modifiedCommunicator.mySend(mV, rank + 1)
            elif rank == 1:
                #Need to recieve mV from party 1
                mV = modifiedCommunicator.myRecieve(rank-1)
                self.share[i] = mV

            mV = self.jmpSend(mV, mV, 0, 1, 2, rank)
            self.share[i] = mV

        self.onlineCount = self.onlineCount + 1
                
    #Function will generate 2 shares per element in the tensor one using the previous key and the other using the next key 
    def getRandomShareForTensor(self, rank, value, index = 0):
        
        #Get shares for all parties
        sharePrev = np.uint64(generate_random_shareFromAES("prev", rank))
        shareNext = np.uint64(generate_random_shareFromAES("next", rank))

        shareThird = self.jmpSend(shareNext, sharePrev,1, 2, 0, rank)
        
     
        return (sharePrev, shareNext, shareThird)

    def privateAddition(self, a, b, iteration):
        if iteration == 0:
            self.masks.append([])
            self.randomShare.append([])
            self.zminusrmask.append([])
            for i in range(0, len(a.masks[a.preprocessingCount - 1])):
                self.masks[self.preprocessingCount].append([(a.masks[a.preprocessingCount - 1][i][0] + b.masks[b.preprocessingCount - 1][i][0]), (a.masks[a.preprocessingCount - 1][i][1] + b.masks[b.preprocessingCount - 1][i][1])])

            self.preprocessingCount = self.preprocessingCount + 1
        else:
            
            for i in range(0, len(self.share)):
                self.share[i] = (a.share[i] + b.share[i])

            self.onlineCount = self.onlineCount + 1


    def publicAddition(self, a, iteration):
        #Adds result of a and b in self
        if iteration == 1:
            for i in range(0, len(self.share)):
                self.share[i] = (self.share[i] + a*self.scale)

    def publicMultiplication(self, a, mult, iteration):
        #Adds result of self and a in self
        mult = np.uint64(mult)
        if iteration == 0:
            self.masks.append([])
            self.randomShare.append([])
            self.zminusrmask.append([])
            for i in range(0, len(a.masks[a.preprocessingCount - 1])):
                self.masks[self.preprocessingCount].append([np.uint64(a.masks[a.preprocessingCount - 1][i][0]*mult), np.uint64(a.masks[a.preprocessingCount - 1][i][1]*mult)])

            self.preprocessingCount = self.preprocessingCount + 1
               
        else:
            
            for i in range(0, len(self.share)):
               
                self.share[i] = np.uint64(a.share[i]*mult)

            self.onlineCount = self.onlineCount + 1
              
    

    def dotPreprocessing(self, rank, a, b, index):
        #Get Fzero functionality
        ra = np.uint64(generate_random_shareFromAES("prev", rank))
        rb = np.uint64(generate_random_shareFromAES("next", rank)) 

        valToAdd = (ra-rb)
 
        share1 = np.uint64(0)
        share2 = np.uint64(0)

        for i in range(0, len(a)):
            share1 = (share1 + a[i][0]*b[i][0] + a[i][0]*b[i][1] + b[i][0]*a[i][1])

        share1 = (share1 + valToAdd)

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

    def truncatedGen(self, rank, index):
        #Number of bits in the number
        l = 64
        x = 16

        #Get shares required for RSS of 4 vectors a, b, c and d
        powersOf2 = []
        cur = np.uint64(1)
        arrSharesb1 = []
        arrSharesb2 = []
        num = np.uint64(0)
        for i in range(0, l):
            powersOf2.append(cur)
            cur *= np.uint64(2)
            b1 = 0, 
            b2 = 0,
            sharesb1 = []
            sharesb2 = []
            if i != l-1:
                if rank == 0:
                    b1 = np.uint64(generate_random_shareFromAES("prev", rank)%2)
                   
                    sharesb1 = [b1, np.uint64(0)]
                    sharesb2 = [np.uint64(0), np.uint64(0)]
                

                elif rank == 1:
                    b2 = np.uint64(generate_random_shareFromAES("next", rank)%2)
                   
                    sharesb1 = [np.uint64(0), np.uint64(0)]
                    sharesb2 = [np.uint64(0), b2]
                
                else:
                    b1 = np.uint64(generate_random_shareFromAES("next", rank)%2)

                    b2 = np.uint64(generate_random_shareFromAES("prev", rank)%2)
                    num = num + powersOf2[i]*(b1^b2)
                
                    sharesb1 = [np.uint64(0), b1]
                    sharesb2 = [b2, np.uint64(0)]
            else:
                if rank == 0:  
                    sharesb1 = [np.uint64(0), np.uint64(0)]
                    sharesb2 = [np.uint64(0), np.uint64(0)]
                
                elif rank == 1:
                    sharesb1 = [np.uint64(0), np.uint64(0)]
                    sharesb2 = [np.uint64(0), np.uint64(0)]
                else:
                    sharesb1 = [np.uint64(0), np.uint64(0)]
                    sharesb2 = [np.uint64(0), np.uint64(0)]
         
            arrSharesb1.append(sharesb1)
            arrSharesb2.append(sharesb2)

        if index == 0 and rank == 2:
            num1 = num
            for i in range(0, 16):
                num1 = int(floor(num1/2))


        powersOf2.append(cur)  
        a = []
        b = []
        c = []
        d = []
        


        #Get the RSS of a b 
        for i in range(0, l):
            a.append([
                (powersOf2[i + 1]*arrSharesb1[i][0]),
                (powersOf2[i + 1]*arrSharesb1[i][1])
            ])
            b.append([
                arrSharesb2[i][0],
                arrSharesb2[i][1]
            ])
        
        #Get the RSS of c d 
        for i in range(0, l - x):
            c.append([
                (powersOf2[i + 1]*arrSharesb1[i + x][0]),
                (powersOf2[i + 1]*arrSharesb1[i + x][1])
            ])
            d.append([
                arrSharesb2[i + x][0],
                arrSharesb2[i + x][1]
            ])
      
        #Get rss shares of a.b and c.d
        res1 = self.dotPreprocessing(rank, a, b, index)
        res2 =  self.dotPreprocessing(rank, c, d, index)
    
        sharesR = []
        sharesRt = []

        sum1 = np.uint64(0)
        sum2 = np.uint64(0)

        for i in range(0, l):
           
            sum1 = (sum1 + (powersOf2[i]*(arrSharesb1[i][0] + arrSharesb2[i][0])))
            sum2 = (sum2 + (powersOf2[i]*(arrSharesb1[i][1] + arrSharesb2[i][1])))
        
        sum1 = (sum1 - res1[0])
        sum2 = (sum2 - res1[1])

        sharesR.append(sum1)
        sharesR.append(sum2)


        sum1 = np.uint64(0)
        sum2 = np.uint64(0)
        varMinusOne = np.uint64(-1)
        for i in range(0, l-x):
            
            
            sum1 = (sum1 + (powersOf2[i]*(varMinusOne*arrSharesb1[i + x][0] + varMinusOne*arrSharesb2[i + x][0])))
            sum2 = (sum2 + (powersOf2[i]*(varMinusOne*arrSharesb1[i + x][1] + varMinusOne*arrSharesb2[i + x][1])))
        
        sum1 = (sum1 + res2[0])
        sum2 = (sum2 + res2[1])

        sharesRt.append(sum1)
        sharesRt.append(sum2)

        return [sharesR, sharesRt]



    #TODO:Replace using ZKP
    def idealPreMult(self, rank, index, a, b):

        share1 = np.uint64(0)
        share2 = np.uint64(0)

        #n - A's masks m - B's masks
        n = a.preprocessingCount - 1
        m = b.preprocessingCount - 1

        #Get first share
        share1 = (a.masks[n][index][0]*b.masks[m][index][0] + a.masks[n][index][0]*b.masks[m][index][1] + b.masks[m][index][0]*a.masks[n][index][1])


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

    def getJointSharingShares(self, rank, index, randomVal1, randomVal2):
      
        share1 = 0, 
        share2 = 0, 
        share3 = 0
        #Here both rank 0 and rank 1 know all three shares
        if rank == 0:
            share1 = np.uint64(0)
            share2 = np.uint64(generate_random_shareFromAES("next", rank))
            share3 = np.uint64(generate_random_shareFromAES("global", rank))
            self.zminusrmask[self.preprocessingCount].append((share1 + share2 + share3))
           
          
        elif rank == 1:
            share1 = np.uint64(generate_random_shareFromAES("prev", rank)) 
            share2 = np.uint64(generate_random_shareFromAES("global", rank)) 
            share3 = np.uint64(0)
            self.zminusrmask[self.preprocessingCount].append((share1 + share2 + share3))
     
       
        else:
            share1 = np.uint64(generate_random_shareFromAES("global", rank)) 
            share2 = np.uint64(0)
            share3 = np.uint64(0)
            self.zminusrmask[self.preprocessingCount].append((0))


        self.zminusrmask[self.preprocessingCount][index] = self.jmpSend(self.zminusrmask[self.preprocessingCount][index], self.zminusrmask[self.preprocessingCount][index], 0, 1, 2, rank)
        self.masks[self.preprocessingCount].append([(share1 + randomVal1), (share2 + randomVal2), np.uint64(0)])
       


    def mainMultiplicationProtocol(self, rank, index, a, b, truncate):
        #Compute ys and jmp share
        n = a.onlineCount - 1
        m = b.onlineCount - 1
        y1 = (np.uint64(-1)*a.masks[n][index][0]*b.share[index] + np.uint64(-1)*b.masks[m][index][0]*a.share[index]+ self.randomShare[self.onlineCount][index][0])
        y2 = (np.uint64(-1)*a.masks[n][index][1]*b.share[index] + np.uint64(-1)*b.masks[m][index][1]*a.share[index]+ self.randomShare[self.onlineCount][index][1])

        ya = self.jmpSend(y2, y1, 1, 2, 0, rank)
        yb = self.jmpSend(y1, y2, 0, 2, 1, rank)
     
        nety = 0
        if rank == 0:
        
            y3 = ya
            nety = (y1 + y2 + y3 + a.share[index]*b.share[index])
         
            if nety >= 2**63:
                nety = int(nety - 2**64)
            else:
                nety = int(nety)

            nety = nety >> 16
            if nety < 0:
                nety = np.uint64(nety + 2**64)
            else:
                nety = np.uint64(nety)
            self.tempShares[index] = (nety + self.zminusrmask[self.onlineCount][index])

        elif rank == 1:
    
            y3 = yb
            nety = (y1 + y2 + y3 + a.share[index]*b.share[index])
          

            if nety >= 2**63:
                nety = int(nety - 2**64)
            else:
                nety = int(nety)

            nety = nety >> 16

            if nety < 0:
                nety = np.uint64(nety + 2**64)
            else:
                nety = np.uint64(nety)

            self.tempShares[index] = (nety + self.zminusrmask[self.onlineCount][index])


        #Send nety to party 2
        nety = self.jmpSend(nety, nety, 0, 1, 2, rank)
        
        self.tempShares[index] = (nety + self.zminusrmask[self.onlineCount][index])
       

    def privateMultiplicationTruncation(self, a, b, iteration):
        rank = comm.get().rank
       

        if iteration == 0:
            #Update preprocessing count and append masks and random shares
            self.masks.append([])
            self.randomShare.append([])
            self.zminusrmask.append([])

            #Need to get gamma shares and r shares stored. Apart from that, joint sharing shares also needed
            for i in range(0, len(a.masks[a.preprocessingCount - 1])):
                res = self.idealPreMult(rank, i, a, b)
                share1 = res[0]
                share2 = res[1]

                res = self.truncatedGen(rank, i)
                self.randomShare[self.preprocessingCount].append([(share1 - res[0][0]), (share2 - res[0][1])])
                self.getJointSharingShares(rank, i, res[1][0], res[1][1])
            
            self.preprocessingCount = self.preprocessingCount + 1

        else:
            self.tempShares = [np.uint64(0) for i in range(0, len(a.share))]
            for i in range(0, len(a.masks[a.onlineCount - 1])):          
                self.mainMultiplicationProtocol(rank, i, a, b, 1)
            self.share = []
            self.share = [self.tempShares[i] for i in range(0, len(self.tempShares))]
            self.onlineCount = self.onlineCount + 1
                
            

    #Functions for preprocessing
    def secretSharingPreprocessing(self, tensor, rank):
        i = 0
        self.masks.append([])
        self.randomShare.append([])
        self.zminusrmask.append([])
        for val in tensor:
            (sharePrev, shareNext, shareThird) = self.getRandomShareForTensor(rank, tensor, i)

            self.masks[self.preprocessingCount].append([sharePrev, shareNext, shareThird])
            
            i = i + 1
        self.preprocessingCount = self.preprocessingCount + 1

        
