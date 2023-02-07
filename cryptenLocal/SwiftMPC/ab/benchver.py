from base64 import encode
from math import floor
import sys
import warnings

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
import cryptenLocal.SwiftMPC as swift
from cryptenLocal.SwiftMPC.ouputCommitments import getFullCommitmentHash, convertHexHashToInteger
from cryptenLocal.SwiftMPC.primitives import boolean

import ouputCommitments
from ab.shuffle import *
from ab.modifiedMPC import *
import time

#To ignore overflow warnings
warnings.filterwarnings("ignore")

p = 340282366920938463463374607431768211297

#Parties 0, 1, 2 computing parties and other parties are client
@swift.run_multiprocess(world_size=3)
def vbenny(numBlocks=1, numClients=10):
    rank = comm.get().rank
    world_size = comm.get().world_size
    # numBlocks = 1 
    # numClients = 10


    if rank == 0:
        al0, al1 = generateRandomness(rank, numBlocks, numClients)
        pi0 = getRandomPermutation(rank, rank, 2, al0)
        pi1 = getRandomPermutation(rank, rank, 0, al1)
        k0, k1 = generateRandomness(rank, 1, numClients)

        sentStart = modifiedCommunicator.totalBitsSent
        sentStart0 = modifiedCommunicator.numberBitsSent0
        sentStart1 = modifiedCommunicator.numberBitsSent1
        sentStart2 = modifiedCommunicator.numberBitsSent2
        recievedStart = modifiedCommunicator.totalBitsRecieved
        recievedStart0 = modifiedCommunicator.numberBitsRecieved0
        recievedStart1 = modifiedCommunicator.numberBitsRecieved1
        recievedStart2 = modifiedCommunicator.numberBitsRecieved2

        timeStart = time.time()

        al0 = list(np.concatenate((np.array(al0), np.array(k0)), axis=1))
        al1 = list(np.concatenate((np.array(al1), np.array(k1)), axis=1))

        al0_, al1_ = al0, al1
        set_equality(rank, numBlocks, len(al0), al0, al1, al0_, al1_)
        timeEnd = time.time()

        sentEnd = modifiedCommunicator.totalBitsSent
        sentEnd0 = modifiedCommunicator.numberBitsSent0
        sentEnd1 = modifiedCommunicator.numberBitsSent1
        sentEnd2 = modifiedCommunicator.numberBitsSent2
        recievedEnd = modifiedCommunicator.totalBitsRecieved
        recievedEnd0 = modifiedCommunicator.numberBitsRecieved0
        recievedEnd1 = modifiedCommunicator.numberBitsRecieved1
        recievedEnd2 = modifiedCommunicator.numberBitsRecieved2
        totime =  timeEnd - timeStart
        totalCommSent = sentEnd - sentStart
        totalCommReceived = recievedEnd - recievedStart
        totalCommSent0 = sentEnd0 - sentStart0
        totalCommReceived0 = recievedEnd0 - recievedStart0
        totalCommSent1 = sentEnd1 - sentStart1
        totalCommReceived1 = recievedEnd1 - recievedStart1
        totalCommSent2 = sentEnd2 - sentStart2
        totalCommReceived2 = recievedEnd2 - recievedStart2
        f = open("logs/benny/vparty0.txt", "a")
        f.write("\n--------------------------------------\n--------------------------------------\n")
        f.write(f"NumClients: {numClients}\n")
        f.write(f"MsgSize: {numBlocks*16}\n")
        f.write("---------------Stats------------------\n")
        f.write(f"My time: {totime}\n")
        f.write(f"Sent to 0: {totalCommSent0*3}\n")
        f.write(f"Sent to 1: {totalCommSent1*3}\n")
        f.write(f"Sent to 2: {totalCommSent2*3}\n")
        f.write(f"Received from 0: {totalCommReceived0*3}\n")
        f.write(f"Received from 1: {totalCommReceived1*3}\n")
        f.write(f"Received from 2: {totalCommReceived2*3}\n")
        f.write(f'total sent: {totalCommSent*3}\n')
        f.write(f'total reeived: {totalCommReceived*3}\n')
        f.write("--------------------------------------\n--------------------------------------\n\n")
    
    if rank == 1:
        al1, al2 = generateRandomness(rank, numBlocks, numClients)
        pi1 = getRandomPermutation(rank, rank, 0, al1)
        pi2 = getRandomPermutation(rank, rank, 2, al2)
        k1, k2 = generateRandomness(rank, 1, numClients)

        sentStart = modifiedCommunicator.totalBitsSent
        sentStart0 = modifiedCommunicator.numberBitsSent0
        sentStart1 = modifiedCommunicator.numberBitsSent1
        sentStart2 = modifiedCommunicator.numberBitsSent2
        recievedStart = modifiedCommunicator.totalBitsRecieved
        recievedStart0 = modifiedCommunicator.numberBitsRecieved0
        recievedStart1 = modifiedCommunicator.numberBitsRecieved1
        recievedStart2 = modifiedCommunicator.numberBitsRecieved2
        timeStart = time.time()

        al1 = list(np.concatenate((np.array(al1), np.array(k1)), axis=1))
        al2 = list(np.concatenate((np.array(al2), np.array(k2)), axis=1))

        #r1
        al1_, al2_ = al1, al2
        set_equality(rank, numBlocks, len(al1), al1, al2, al1_, al2_)

        timeEnd = time.time()
        sentEnd = modifiedCommunicator.totalBitsSent
        sentEnd0 = modifiedCommunicator.numberBitsSent0
        sentEnd1 = modifiedCommunicator.numberBitsSent1
        sentEnd2 = modifiedCommunicator.numberBitsSent2
        recievedEnd = modifiedCommunicator.totalBitsRecieved
        recievedEnd0 = modifiedCommunicator.numberBitsRecieved0
        recievedEnd1 = modifiedCommunicator.numberBitsRecieved1
        recievedEnd2 = modifiedCommunicator.numberBitsRecieved2
        totime =  timeEnd - timeStart
        totalCommSent = sentEnd - sentStart
        totalCommReceived = recievedEnd - recievedStart
        totalCommSent0 = sentEnd0 - sentStart0
        totalCommReceived0 = recievedEnd0 - recievedStart0
        totalCommSent1 = sentEnd1 - sentStart1
        totalCommReceived1 = recievedEnd1 - recievedStart1
        totalCommSent2 = sentEnd2 - sentStart2
        totalCommReceived2 = recievedEnd2 - recievedStart2
        f = open("logs/benny/vparty1.txt", "a")
        f.write("\n--------------------------------------\n--------------------------------------\n")
        f.write(f"NumClients: {numClients}\n")
        f.write(f"MsgSize: {numBlocks*16}\n")
        f.write("---------------Stats------------------\n")
        f.write(f"My time: {totime}\n")
        f.write(f"Sent to 0: {totalCommSent0*3}\n")
        f.write(f"Sent to 1: {totalCommSent1*3}\n")
        f.write(f"Sent to 2: {totalCommSent2*3}\n")
        f.write(f"Received from 0: {totalCommReceived0*3}\n")
        f.write(f"Received from 1: {totalCommReceived1*3}\n")
        f.write(f"Received from 2: {totalCommReceived2*3}\n")
        f.write(f'total sent: {totalCommSent*3}\n')
        f.write(f'total reeived: {totalCommReceived*3}\n')
        f.write("--------------------------------------\n--------------------------------------\n\n")
        

    if rank == 2:
        al2, al0 = generateRandomness(rank, numBlocks, numClients)
        pi2 = getRandomPermutation(rank, rank, 1, al2)
        pi0 = getRandomPermutation(rank, rank, 0, al0)
        k2, k0 = generateRandomness(rank, 1, numClients)

        sentStart = modifiedCommunicator.totalBitsSent
        sentStart0 = modifiedCommunicator.numberBitsSent0
        sentStart1 = modifiedCommunicator.numberBitsSent1
        sentStart2 = modifiedCommunicator.numberBitsSent2
        recievedStart = modifiedCommunicator.totalBitsRecieved
        recievedStart0 = modifiedCommunicator.numberBitsRecieved0
        recievedStart1 = modifiedCommunicator.numberBitsRecieved1
        recievedStart2 = modifiedCommunicator.numberBitsRecieved2
        timeStart = time.time()


        al2 = list(np.concatenate((np.array(al2), np.array(k2)), axis=1))
        al0 = list(np.concatenate((np.array(al0), np.array(k0)), axis=1))

        #r1
        al2_, al0_ = al2, al0
        set_equality(rank, numBlocks, len(al2), al2, al0, al2_, al0_)

        timeEnd = time.time()
        sentEnd = modifiedCommunicator.totalBitsSent
        sentEnd0 = modifiedCommunicator.numberBitsSent0
        sentEnd1 = modifiedCommunicator.numberBitsSent1
        sentEnd2 = modifiedCommunicator.numberBitsSent2
        recievedEnd = modifiedCommunicator.totalBitsRecieved
        recievedEnd0 = modifiedCommunicator.numberBitsRecieved0
        recievedEnd1 = modifiedCommunicator.numberBitsRecieved1
        recievedEnd2 = modifiedCommunicator.numberBitsRecieved2
        totime =  timeEnd - timeStart
        totalCommSent = sentEnd - sentStart
        totalCommReceived = recievedEnd - recievedStart
        totalCommSent0 = sentEnd0 - sentStart0
        totalCommReceived0 = recievedEnd0 - recievedStart0
        totalCommSent1 = sentEnd1 - sentStart1
        totalCommReceived1 = recievedEnd1 - recievedStart1
        totalCommSent2 = sentEnd2 - sentStart2
        totalCommReceived2 = recievedEnd2 - recievedStart2
        f = open("logs/benny/vparty2.txt", "a")
        f.write("\n--------------------------------------\n--------------------------------------\n")
        f.write(f"NumClients: {numClients}\n")
        f.write(f"MsgSize: {numBlocks*16}\n")
        f.write("---------------Stats------------------\n")
        f.write(f"My time: {totime}\n")
        f.write(f"Sent to 0: {totalCommSent0*3}\n")
        f.write(f"Sent to 1: {totalCommSent1*3}\n")
        f.write(f"Sent to 2: {totalCommSent2*3}\n")
        f.write(f"Received from 0: {totalCommReceived0*3}\n")
        f.write(f"Received from 1: {totalCommReceived1*3}\n")
        f.write(f"Received from 2: {totalCommReceived2*3}\n")
        f.write(f'total sent: {totalCommSent*3}\n')
        f.write(f'total reeived: {totalCommReceived*3}\n')
        f.write("--------------------------------------\n--------------------------------------\n\n")

print("1")
vbenny(numBlocks=1, numClients=100)
print("2")
vbenny(numBlocks=1, numClients=1000)
print("3")
vbenny(numBlocks=1, numClients=10000)
print("4")
vbenny(numBlocks=1, numClients=10000)
