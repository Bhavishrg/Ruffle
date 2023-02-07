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
from ecdsa import SigningKey


#To ignore overflow warnings
warnings.filterwarnings("ignore")

p = 340282366920938463463374607431768211297

#Parties 0, 1, 2 computing parties and other parties are client
@swift.run_multiprocess(world_size=3)
def swiftShuffle(numBlocks, numClients):
    rank = comm.get().rank
    world_size = comm.get().world_size
    # numBlocks = 1 
    # numClients = 10
    
    MtimeS = time.time()

    if rank == 0:
        ## Generate required randomness ##
        al0, al1 = generateRandomness(rank, numBlocks, numClients)
        r0, r1 = generateRandomness(rank, numBlocks, numClients)
        k0, k1 = generateRandomness(rank, 1, numClients)
        pi0 = getRandomPermutation(rank, rank, 2, al0)
        pi1 = getRandomPermutation(rank, rank, 0, al1)


        sentStart = modifiedCommunicator.totalBitsSent
        sentStart0 = modifiedCommunicator.numberBitsSent0
        sentStart1 = modifiedCommunicator.numberBitsSent1
        sentStart2 = modifiedCommunicator.numberBitsSent2
        recievedStart = modifiedCommunicator.totalBitsRecieved
        recievedStart0 = modifiedCommunicator.numberBitsRecieved0
        recievedStart1 = modifiedCommunicator.numberBitsRecieved1
        recievedStart2 = modifiedCommunicator.numberBitsRecieved2
        timeStart = time.time()

        ### Preprocessing Phase ###
        al_0, al_1 = shufflePre(rank, numBlocks, al0, al1, r0, r1, pi0, pi1, k0, k1)

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
        f = open("logs/shuffle/party0.txt", "a")
        f.write("\n--------------------------------------\n--------------------------------------\n")
        f.write("\n--------------------------------------\n--------------------------------------\n")
        f.write(f"NumClients: {numClients}\n")
        f.write(f"MsgSize: {numBlocks*16}\n")
        f.write("---------------Preprocessing------------------\n")
        f.write(f"My time: {totime}\n")
        f.write(f"Sent to 0: {totalCommSent0}\n")
        f.write(f"Sent to 1: {totalCommSent1}\n")
        f.write(f"Sent to 2: {totalCommSent2}\n")
        f.write(f"Received from 0: {totalCommReceived0}\n")
        f.write(f"Received from 1: {totalCommReceived1}\n")
        f.write(f"Received from 2: {totalCommReceived2}\n")
        f.write(f'total sent: {totalCommSent}\n')
        f.write(f'total reeived: {totalCommReceived}\n')
        f.write("--------------------------------------\n--------------------------------------\n\n")

        ### Online phase ###
        beta = list(np.ones((numClients, numBlocks)))
        ## generateBeta ##
        sentStart = modifiedCommunicator.totalBitsSent
        sentStart0 = modifiedCommunicator.numberBitsSent0
        sentStart1 = modifiedCommunicator.numberBitsSent1
        sentStart2 = modifiedCommunicator.numberBitsSent2
        recievedStart = modifiedCommunicator.totalBitsRecieved
        recievedStart0 = modifiedCommunicator.numberBitsRecieved0
        recievedStart1 = modifiedCommunicator.numberBitsRecieved1
        recievedStart2 = modifiedCommunicator.numberBitsRecieved2
        timeStart = time.time()
        print(rank, r0)
        beta_ = ShuffleOnl(rank, numBlocks, beta, r0, r1, pi0, pi1)

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
        f = open("logs/shuffle/party0.txt", "a")
        f.write("\n--------------------------------------\n--------------------------------------\n")
        f.write("\n--------------------------------------\n--------------------------------------\n")
        f.write(f"NumClients: {numClients}\n")
        f.write(f"MsgSize: {numBlocks*16}\n")
        f.write("---------------Online------------------\n")
        f.write(f"My time: {totime}\n")
        f.write(f"Sent to 0: {totalCommSent0}\n")
        f.write(f"Sent to 1: {totalCommSent1}\n")
        f.write(f"Sent to 2: {totalCommSent2}\n")
        f.write(f"Received from 0: {totalCommReceived0}\n")
        f.write(f"Received from 1: {totalCommReceived1}\n")
        f.write(f"Received from 2: {totalCommReceived2}\n")
        f.write(f'total sent: {totalCommSent}\n')
        f.write(f'total reeived: {totalCommReceived}\n')
        f.write("--------------------------------------\n--------------------------------------\n\n")
    
    if rank == 1:

        ### Preprocessing Phase ###
        ## Generate required randomness ##
        al1, al2 = generateRandomness(rank, numBlocks, numClients)
        r1, r2 = generateRandomness(rank, numBlocks, numClients)
        k1, k2 = generateRandomness(rank, 1, numClients)
        pi1 = getRandomPermutation(rank, rank, 0, al1)
        pi2 = getRandomPermutation(rank, rank, 2, al2)

        ## Preprocessing shuffle ##

        sentStart = modifiedCommunicator.totalBitsSent
        sentStart0 = modifiedCommunicator.numberBitsSent0
        sentStart1 = modifiedCommunicator.numberBitsSent1
        sentStart2 = modifiedCommunicator.numberBitsSent2
        recievedStart = modifiedCommunicator.totalBitsRecieved
        recievedStart0 = modifiedCommunicator.numberBitsRecieved0
        recievedStart1 = modifiedCommunicator.numberBitsRecieved1
        recievedStart2 = modifiedCommunicator.numberBitsRecieved2

        timeStart = time.time()

        al_1, al_2 = shufflePre(rank, numBlocks, al1, al2, r1, r2, pi1, pi2, k1, k2)

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
        f = open("logs/shuffle/party1.txt", "a")
        f.write("\n--------------------------------------\n--------------------------------------\n")
        f.write("\n--------------------------------------\n--------------------------------------\n")
        f.write(f"NumClients: {numClients}\n")
        f.write(f"MsgSize: {numBlocks*16}\n")
        f.write("---------------Preproessing------------------\n")
        f.write(f"My time: {totime}\n")
        f.write(f"Sent to 0: {totalCommSent0}\n")
        f.write(f"Sent to 1: {totalCommSent1}\n")
        f.write(f"Sent to 2: {totalCommSent2}\n")
        f.write(f"Received from 0: {totalCommReceived0}\n")
        f.write(f"Received from 1: {totalCommReceived1}\n")
        f.write(f"Received from 2: {totalCommReceived2}\n")
        f.write(f'total sent: {totalCommSent}\n')
        f.write(f'total reeived: {totalCommReceived}\n')
        f.write("--------------------------------------\n--------------------------------------\n\n")
        
        ### Online phase ###
        ## generateBeta ##
        beta = list(np.ones((numClients, numBlocks)))

        sentStart = modifiedCommunicator.totalBitsSent
        sentStart0 = modifiedCommunicator.numberBitsSent0
        sentStart1 = modifiedCommunicator.numberBitsSent1
        sentStart2 = modifiedCommunicator.numberBitsSent2
        recievedStart = modifiedCommunicator.totalBitsRecieved
        recievedStart0 = modifiedCommunicator.numberBitsRecieved0
        recievedStart1 = modifiedCommunicator.numberBitsRecieved1
        recievedStart2 = modifiedCommunicator.numberBitsRecieved2
        timeStart = time.time()

        beta_ = ShuffleOnl(rank, numBlocks,  beta, r1, r2, pi1, pi2)

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
        f = open("logs/shuffle/party1.txt", "a")
        f.write("\n--------------------------------------\n--------------------------------------\n")
        f.write("\n--------------------------------------\n--------------------------------------\n")
        f.write(f"NumClients: {numClients}\n")
        f.write(f"MsgSize: {numBlocks*16}\n")
        f.write("---------------StOnlineats------------------\n")
        f.write(f"My time: {totime}\n")
        f.write(f"Sent to 0: {totalCommSent0}\n")
        f.write(f"Sent to 1: {totalCommSent1}\n")
        f.write(f"Sent to 2: {totalCommSent2}\n")
        f.write(f"Received from 0: {totalCommReceived0}\n")
        f.write(f"Received from 1: {totalCommReceived1}\n")
        f.write(f"Received from 2: {totalCommReceived2}\n")
        f.write(f'total sent: {totalCommSent}\n')
        f.write(f'total reeived: {totalCommReceived}\n')
        f.write("--------------------------------------\n--------------------------------------\n\n")


    if rank == 2:
       
        ### Preprocessing Phase ###
        ## Generate required randomness ##
        al2, al0 = generateRandomness(rank, numBlocks, numClients)
        r2, r0 = generateRandomness(rank, numBlocks, numClients)
        k2, k0 = generateRandomness(rank, 1, numClients)
        pi2 = getRandomPermutation(rank, rank, 1, al2)
        pi0 = getRandomPermutation(rank, rank, 0, al0)

        sentStart = modifiedCommunicator.totalBitsSent
        sentStart0 = modifiedCommunicator.numberBitsSent0
        sentStart1 = modifiedCommunicator.numberBitsSent1
        sentStart2 = modifiedCommunicator.numberBitsSent2
        recievedStart = modifiedCommunicator.totalBitsRecieved
        recievedStart0 = modifiedCommunicator.numberBitsRecieved0
        recievedStart1 = modifiedCommunicator.numberBitsRecieved1
        recievedStart2 = modifiedCommunicator.numberBitsRecieved2

        timeStart = time.time()

        al_2, al_0 = shufflePre(rank, numBlocks, al2, al0, r2, r0, pi2, pi0, k2, k0)

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
        f = open("logs/shuffle/party2.txt", "a")
        f.write("\n--------------------------------------\n--------------------------------------\n")
        f.write("\n--------------------------------------\n--------------------------------------\n")
        f.write(f"NumClients: {numClients}\n")
        f.write(f"MsgSize: {numBlocks*16}\n")
        f.write("---------------Preprocessing------------------\n")
        f.write(f"My time: {totime}\n")
        f.write(f"Sent to 0: {totalCommSent0}\n")
        f.write(f"Sent to 1: {totalCommSent1}\n")
        f.write(f"Sent to 2: {totalCommSent2}\n")
        f.write(f"Received from 0: {totalCommReceived0}\n")
        f.write(f"Received from 1: {totalCommReceived1}\n")
        f.write(f"Received from 2: {totalCommReceived2}\n")
        f.write(f'total sent: {totalCommSent}\n')
        f.write(f'total reeived: {totalCommReceived}\n')
        f.write("--------------------------------------\n--------------------------------------\n\n")

        ### Online phase ###
        ## generateBeta ##
        beta = list(np.ones((numClients, numBlocks)))

        sentStart = modifiedCommunicator.totalBitsSent
        sentStart0 = modifiedCommunicator.numberBitsSent0
        sentStart1 = modifiedCommunicator.numberBitsSent1
        sentStart2 = modifiedCommunicator.numberBitsSent2
        recievedStart = modifiedCommunicator.totalBitsRecieved
        recievedStart0 = modifiedCommunicator.numberBitsRecieved0
        recievedStart1 = modifiedCommunicator.numberBitsRecieved1
        recievedStart2 = modifiedCommunicator.numberBitsRecieved2
        timeStart = time.time()

        beta_ = ShuffleOnl(rank, numBlocks, beta, r2, r0, pi2, pi0)
        
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
        f = open("logs/shuffle/party2.txt", "a")
        f.write("\n--------------------------------------\n--------------------------------------\n")
        f.write("\n--------------------------------------\n--------------------------------------\n")
        f.write(f"NumClients: {numClients}\n")
        f.write(f"MsgSize: {numBlocks*16}\n")
        f.write("---------------Online------------------\n")
        f.write(f"My time: {totime}\n")
        f.write(f"Sent to 0: {totalCommSent0}\n")
        f.write(f"Sent to 1: {totalCommSent1}\n")
        f.write(f"Sent to 2: {totalCommSent2}\n")
        f.write(f"Received from 0: {totalCommReceived0}\n")
        f.write(f"Received from 1: {totalCommReceived1}\n")
        f.write(f"Received from 2: {totalCommReceived2}\n")
        f.write(f'total sent: {totalCommSent}\n')
        f.write(f'total reeived: {totalCommReceived}\n')
        f.write("--------------------------------------\n--------------------------------------\n\n")

    MtimeE = time.time()
    Mtime = MtimeE - MtimeS
    f.write(f'Mtime: {Mtime}\n')
    f.write("--------------------------------------\n--------------------------------------\n\n")