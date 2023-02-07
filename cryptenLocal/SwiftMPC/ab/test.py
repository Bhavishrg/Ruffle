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
@swift.run_multiprocess(world_size=4)
def ab(numBlocks, numClients):
    rank = comm.get().rank
    world_size = comm.get().world_size
    # numBlocks = 1 
    # numClients = 10

    if rank == 0:

        sentStart = modifiedCommunicator.totalBitsSent
        sentStart0 = modifiedCommunicator.numberBitsSent0
        sentStart1 = modifiedCommunicator.numberBitsSent1
        sentStart2 = modifiedCommunicator.numberBitsSent2
        recievedStart = modifiedCommunicator.totalBitsRecieved
        recievedStart0 = modifiedCommunicator.numberBitsRecieved0
        recievedStart1 = modifiedCommunicator.numberBitsRecieved1
        recievedStart2 = modifiedCommunicator.numberBitsRecieved2
        timeStart = time.time()
        ### PKE ###
        sk0 = SigningKey.generate()
        vk0 = sk0.verifying_key
        vk0_ = vk0.to_string().hex()
        vk0_ = [int(vk0_[:16],16), int(vk0_[16:32],16), int(vk0_[32:48],16), int(vk0_[48:64],16), int(vk0_[64:80],16), int(vk0_[80:96],16)]
        vk1 = recoverKey(modifiedCommunicator.myRecieve1(1, [1,6])[0])
        modifiedCommunicator.mySend1(vk0_, 1)
        modifiedCommunicator.mySend1(vk0_, 2)
        vk2 = recoverKey(modifiedCommunicator.myRecieve1(2, [1,6])[0])

        ## Generate required randomness ##
        al0, al1 = generateRandomness(rank, numBlocks, numClients)
        r0, r1 = generateRandomness(rank, numBlocks, numClients)
        k0, k1 = generateRandomness(rank, 1, numClients)
        d0, d1 = generateRandomness(rank, 1, numClients)
        pi0 = getRandomPermutation(rank, rank, 2, al0)
        pi1 = getRandomPermutation(rank, rank, 0, al1)
        q0, q1 = generateRandomness(rank, 1, numClients)

        # exchange commitments #
        c = commitments(rank, al0, al1)

        ### Preprocessing Phase ###
        al_0, al_1 = shufflePre(rank, numBlocks, al0, al1, r0, r1, pi0, pi1, k0, k1)

        ##Output commitments ##
        cal = commitments(rank, al_0, al_1)

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

        print("logs/ab/party0.txt", "a")
        print("\n--------------------------------------\n--------------------------------------\n")
        print(f"NumClients: {numClients}\n")
        print(f"MsgSize: {numBlocks*16}\n")
        print("---------------Preprocessing------------------\n")
        print(f"My time: {totime}\n")
        print(f"Sent to 0: {totalCommSent0}\n")
        print(f"Sent to 1: {totalCommSent1}\n")
        print(f"Sent to 2: {totalCommSent2}\n")
        print(f"Received from 0: {totalCommReceived0}\n")
        print(f"Received from 1: {totalCommReceived1}\n")
        print(f"Received from 2: {totalCommReceived2}\n")
        print(f'total sent: {totalCommSent}\n')
        print(f'total reeived: {totalCommReceived}\n')
        print("--------------------------------------\n--------------------------------------\n\n")

        ## send alpha and its commitments to clients ##
        sendtoclients(rank, al0, al1, c)
        ## generateBeta ##
        beta = generateBeta(rank, numBlocks, numClients)

        modifiedCommunicator.totalBitsSent = 0
        modifiedCommunicator.numberBitsSent0 = 0
        modifiedCommunicator.numberBitsSent1 = 0
        modifiedCommunicator.numberBitsSent2 = 0
        modifiedCommunicator.totalBitsRecieved = 0
        modifiedCommunicator.numberBitsRecieved0 = 0
        modifiedCommunicator.numberBitsRecieved1 = 0
        modifiedCommunicator.numberBitsRecieved2 = 0
        sentStart = modifiedCommunicator.totalBitsSent
        sentStart0 = modifiedCommunicator.numberBitsSent0
        sentStart1 = modifiedCommunicator.numberBitsSent1
        sentStart2 = modifiedCommunicator.numberBitsSent2
        recievedStart = modifiedCommunicator.totalBitsRecieved
        recievedStart0 = modifiedCommunicator.numberBitsRecieved0
        recievedStart1 = modifiedCommunicator.numberBitsRecieved1
        recievedStart2 = modifiedCommunicator.numberBitsRecieved2
        timeStart = time.time()

        broadcast_ints(rank, beta, sk0, vk0, vk1, vk2)

        beta_ = ShuffleOnl(rank, numBlocks, beta, r0, r1, pi0, pi1)
        reconstOutput(rank, beta_, al_0, al_1)

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
        print("logs/ab/party0.txt", "a")
        print("\n--------------------------------------\n--------------------------------------\n")
        print(f"NumClients: {numClients}\n")
        print(f"MsgSize: {numBlocks*16}\n")
        print("---------------ONLINE------------------\n")
        print(f"My time: {totime}\n")
        print(f"Sent to 0: {totalCommSent0}\n")
        print(f"Sent to 1: {totalCommSent1}\n")
        print(f"Sent to 2: {totalCommSent2}\n")
        print(f"Received from 0: {totalCommReceived0}\n")
        print(f"Received from 1: {totalCommReceived1}\n")
        print(f"Received from 2: {totalCommReceived2}\n")
        print(f'total sent: {totalCommSent}\n')
        print(f'total reeived: {totalCommReceived}\n')
        print("--------------------------------------\n--------------------------------------\n\n")


    if rank == 1:
        sentStart = modifiedCommunicator.totalBitsSent
        sentStart0 = modifiedCommunicator.numberBitsSent0
        sentStart1 = modifiedCommunicator.numberBitsSent1
        sentStart2 = modifiedCommunicator.numberBitsSent2
        recievedStart = modifiedCommunicator.totalBitsRecieved
        recievedStart0 = modifiedCommunicator.numberBitsRecieved0
        recievedStart1 = modifiedCommunicator.numberBitsRecieved1
        recievedStart2 = modifiedCommunicator.numberBitsRecieved2
        timeStart = time.time()

        ### PKE ###
        sk1 = SigningKey.generate()
        vk1 = sk1.verifying_key
        vk1_ = vk1.to_string().hex()
        vk1_ = [int(vk1_[:16],16), int(vk1_[16:32],16), int(vk1_[32:48],16), int(vk1_[48:64],16), int(vk1_[64:80],16), int(vk1_[80:96],16)]
        modifiedCommunicator.mySend1(vk1_, 2)
        modifiedCommunicator.mySend1(vk1_, 0)
        vk0 = recoverKey(modifiedCommunicator.myRecieve1(0, [1,6])[0])
        vk2 = recoverKey(modifiedCommunicator.myRecieve1(2, [1,6])[0])

        ### Preprocessing Phase ###
        ## Generate required randomness ##
        al1, al2 = generateRandomness(rank, numBlocks, numClients)
        r1, r2 = generateRandomness(rank, numBlocks, numClients)
        k1, k2 = generateRandomness(rank, 1, numClients)
        d1, d2 = generateRandomness(rank, 1, numClients)
        pi1 = getRandomPermutation(rank, rank, 0, al1)
        pi2 = getRandomPermutation(rank, rank, 2, al2)
        q1, q2 = generateRandomness(rank, 1, numClients)

        # exchange commitments #
        c = commitments(rank, al1, al2)

        ## Preprocessing shuffle ##
        al_1, al_2 = shufflePre(rank, numBlocks, al1, al2, r1, r2, pi1, pi2, k1, k2)


        ## output commitments ##
        cal = commitments(rank, al_1, al_2)

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
        print("logs/ab/party1.txt", "a")
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

        ## send alpha and its commitments to clients ##
        sendtoclients(rank, al1, al2, c)
        ## generateBeta ##
        beta = generateBeta(rank, numBlocks, numClients)

        modifiedCommunicator.totalBitsSent = 0
        modifiedCommunicator.numberBitsSent0 = 0
        modifiedCommunicator.numberBitsSent1 = 0
        modifiedCommunicator.numberBitsSent2 = 0
        modifiedCommunicator.totalBitsRecieved = 0
        modifiedCommunicator.numberBitsRecieved0 = 0
        modifiedCommunicator.numberBitsRecieved1 = 0
        modifiedCommunicator.numberBitsRecieved2 = 0
        sentStart = modifiedCommunicator.totalBitsSent
        sentStart0 = modifiedCommunicator.numberBitsSent0
        sentStart1 = modifiedCommunicator.numberBitsSent1
        sentStart2 = modifiedCommunicator.numberBitsSent2
        recievedStart = modifiedCommunicator.totalBitsRecieved
        recievedStart0 = modifiedCommunicator.numberBitsRecieved0
        recievedStart1 = modifiedCommunicator.numberBitsRecieved1
        recievedStart2 = modifiedCommunicator.numberBitsRecieved2
        timeStart = time.time()

        broadcast_ints(rank, beta, sk1, vk0, vk1, vk2)
        beta_ = ShuffleOnl(rank, numBlocks,  beta, r1, r2, pi1, pi2)
        reconstOutput(rank, beta_, al_1, al_2)

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
        print("logs/ab/party1.txt", "a")
        print("\n--------------------------------------\n--------------------------------------\n")
        print(f"NumClients: {numClients}\n")
        print(f"MsgSize: {numBlocks*16}\n")
        print("---------------Online------------------\n")
        print(f"My time: {totime}\n")
        print(f"Sent to 0: {totalCommSent0}\n")
        print(f"Sent to 1: {totalCommSent1}\n")
        print(f"Sent to 2: {totalCommSent2}\n")
        print(f"Received from 0: {totalCommReceived0}\n")
        print(f"Received from 1: {totalCommReceived1}\n")
        print(f"Received from 2: {totalCommReceived2}\n")
        print(f'total sent: {totalCommSent}\n')
        print(f'total reeived: {totalCommReceived}\n')
        print("--------------------------------------\n--------------------------------------\n\n")


    if rank == 2:
        sentStart = modifiedCommunicator.totalBitsSent
        sentStart0 = modifiedCommunicator.numberBitsSent0
        sentStart1 = modifiedCommunicator.numberBitsSent1
        sentStart2 = modifiedCommunicator.numberBitsSent2
        recievedStart = modifiedCommunicator.totalBitsRecieved
        recievedStart0 = modifiedCommunicator.numberBitsRecieved0
        recievedStart1 = modifiedCommunicator.numberBitsRecieved1
        recievedStart2 = modifiedCommunicator.numberBitsRecieved2
        timeStart = time.time()

        ### PKE ###
        sk2 = SigningKey.generate()
        vk2 = sk2.verifying_key
        vk2_ = vk2.to_string().hex()
        vk2_ = [int(vk2_[:16],16), int(vk2_[16:32],16), int(vk2_[32:48],16), int(vk2_[48:64],16), int(vk2_[64:80],16), int(vk2_[80:96],16)]
        
        vk1 = recoverKey(modifiedCommunicator.myRecieve1(1, [1,6])[0])
        vk0 = recoverKey(modifiedCommunicator.myRecieve1(0, [1,6])[0])
        
        modifiedCommunicator.mySend1(vk2_, 0)
        modifiedCommunicator.mySend1(vk2_, 1)

        ### Preprocessing Phase ###
        ## Generate required randomness ##
        al2, al0 = generateRandomness(rank, numBlocks, numClients)
        r2, r0 = generateRandomness(rank, numBlocks, numClients)
        k2, k0 = generateRandomness(rank, 1, numClients)
        d2, d0 = generateRandomness(rank, 1, numClients)
        pi2 = getRandomPermutation(rank, rank, 1, al2)
        pi0 = getRandomPermutation(rank, rank, 0, al0)
        q2, q0 = generateRandomness(rank, 1, numClients)


        # exchange commitments #
        c = commitments(rank, al2, al0)

        ## Preprocessing shuffle ##
        al_2, al_0 = shufflePre(rank, numBlocks, al2, al0, r2, r0, pi2, pi0, k2, k0)
        
        ## output commitments ##
        cal = commitments(rank, al_2, al_0)


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
        print("logs/ab/party2.txt", "a")
        print("\n--------------------------------------\n--------------------------------------\n")
        print(f"NumClients: {numClients}\n")
        print(f"MsgSize: {numBlocks*16}\n")
        print("---------------Preprocessing------------------\n")
        print(f"My time: {totime}\n")
        print(f"Sent to 0: {totalCommSent0}\n")
        print(f"Sent to 1: {totalCommSent1}\n")
        print(f"Sent to 2: {totalCommSent2}\n")
        print(f"Received from 0: {totalCommReceived0}\n")
        print(f"Received from 1: {totalCommReceived1}\n")
        print(f"Received from 2: {totalCommReceived2}\n")
        print(f'total sent: {totalCommSent}\n')
        print(f'total reeived: {totalCommReceived}\n')
        print("--------------------------------------\n--------------------------------------\n\n")

        ## send alpha and its commitments to clients ##
        sendtoclients(rank, al2, al0, c)
        ## generateBeta ##
        beta = generateBeta(rank,  numBlocks, numClients)

        modifiedCommunicator.totalBitsSent = 0
        modifiedCommunicator.numberBitsSent0 = 0
        modifiedCommunicator.numberBitsSent1 = 0
        modifiedCommunicator.numberBitsSent2 = 0
        modifiedCommunicator.totalBitsRecieved = 0
        modifiedCommunicator.numberBitsRecieved0 = 0
        modifiedCommunicator.numberBitsRecieved1 = 0
        modifiedCommunicator.numberBitsRecieved2 = 0
        sentStart = modifiedCommunicator.totalBitsSent
        sentStart0 = modifiedCommunicator.numberBitsSent0
        sentStart1 = modifiedCommunicator.numberBitsSent1
        sentStart2 = modifiedCommunicator.numberBitsSent2
        recievedStart = modifiedCommunicator.totalBitsRecieved
        recievedStart0 = modifiedCommunicator.numberBitsRecieved0
        recievedStart1 = modifiedCommunicator.numberBitsRecieved1
        recievedStart2 = modifiedCommunicator.numberBitsRecieved2
        timeStart = time.time()

        broadcast_ints(rank, beta, sk2, vk0, vk1, vk2)
        beta_ = ShuffleOnl(rank, numBlocks, beta, r2, r0, pi2, pi0)
        reconstOutput(rank, beta_, al_2, al_0)

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
        print("logs/ab/party2.txt", "a")
        print("\n--------------------------------------\n--------------------------------------\n")
        print(f"NumClients: {numClients}\n")
        print(f"MsgSize: {numBlocks*16}\n")
        print("---------------Preprocessing------------------\n")
        print(f"My time: {totime}\n")
        print(f"Sent to 0: {totalCommSent0}\n")
        print(f"Sent to 1: {totalCommSent1}\n")
        print(f"Sent to 2: {totalCommSent2}\n")
        print(f"Received from 0: {totalCommReceived0}\n")
        print(f"Received from 1: {totalCommReceived1}\n")
        print(f"Received from 2: {totalCommReceived2}\n")
        print(f'total sent: {totalCommSent}\n')
        print(f'total reeived: {totalCommReceived}\n')
        print("--------------------------------------\n--------------------------------------\n\n")
        


    if rank==3:
        # msg = generateMsg(rank, numBlocks, numClients)
        modifiedCommunicator.totalBitsSent = 0
        modifiedCommunicator.numberBitsSent0 = 0
        modifiedCommunicator.numberBitsSent1 = 0
        modifiedCommunicator.numberBitsSent2 = 0
        modifiedCommunicator.totalBitsRecieved = 0
        modifiedCommunicator.numberBitsRecieved0 = 0
        modifiedCommunicator.numberBitsRecieved1 = 0
        modifiedCommunicator.numberBitsRecieved2 = 0
        sentStart = modifiedCommunicator.totalBitsSent
        sentStart0 = modifiedCommunicator.numberBitsSent0
        sentStart1 = modifiedCommunicator.numberBitsSent1
        sentStart2 = modifiedCommunicator.numberBitsSent2
        recievedStart = modifiedCommunicator.totalBitsRecieved
        recievedStart0 = modifiedCommunicator.numberBitsRecieved0
        recievedStart1 = modifiedCommunicator.numberBitsRecieved1
        recievedStart2 = modifiedCommunicator.numberBitsRecieved2
        

        totime = clientSim(rank, numBlocks, numClients)

        sentEnd = modifiedCommunicator.totalBitsSent
        sentEnd0 = modifiedCommunicator.numberBitsSent0
        sentEnd1 = modifiedCommunicator.numberBitsSent1
        sentEnd2 = modifiedCommunicator.numberBitsSent2
        recievedEnd = modifiedCommunicator.totalBitsRecieved
        recievedEnd0 = modifiedCommunicator.numberBitsRecieved0
        recievedEnd1 = modifiedCommunicator.numberBitsRecieved1
        recievedEnd2 = modifiedCommunicator.numberBitsRecieved2
        totalCommSent = sentEnd - sentStart
        totalCommReceived = recievedEnd - recievedStart
        totalCommSent0 = sentEnd0 - sentStart0
        totalCommReceived0 = recievedEnd0 - recievedStart0
        totalCommSent1 = sentEnd1 - sentStart1
        totalCommReceived1 = recievedEnd1 - recievedStart1
        totalCommSent2 = sentEnd2 - sentStart2
        totalCommReceived2 = recievedEnd2 - recievedStart2

        print("logs/ab/client.txt", "a")
        print("\n--------------------------------------\n--------------------------------------\n")
        print(f"NumClients: {numClients}\n")
        print(f"MsgSize: {numBlocks*16}\n")
        print("---------------Preprocessing------------------\n")
        print(f"My time: {totime}\n")
        print(f"Sent to 0: {totalCommSent0}\n")
        print(f"Sent to 1: {totalCommSent1}\n")
        print(f"Sent to 2: {totalCommSent2}\n")
        print(f"Received from 0: {totalCommReceived0}\n")
        print(f"Received from 1: {totalCommReceived1}\n")
        print(f"Received from 2: {totalCommReceived2}\n")
        print(f'total sent: {totalCommSent}\n')
        print(f'total reeived: {totalCommReceived}\n')
        print("--------------------------------------\n--------------------------------------\n\n")

ab(2, 10)   