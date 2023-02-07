from __future__ import generators
import sys



sys.path.insert(0, '../../')

from cryptenLocal.clarion import modifiedCommunicator
from ctypes import addressof
import cryptenLocal as crypten
import cryptenLocal.clarion as clarion
import cryptenLocal.communicator as comm
import cryptenLocal.SwiftMPC.primitives.arithmetic as arithmetic
import torch
import sys
import os
import hashlib
import time
import warnings
import numpy as np
import client
import server
import modp

#To ignore overflow warnings
warnings.filterwarnings("ignore")

p = 340282366920938463463374607431768211297

#Parties 0, 1, 2 computing parties and other parties are client
@clarion.run_multiprocess(world_size=4)
def clar(numBlocks, numClients):
    
    rank = comm.get().rank
    world_size = comm.get().world_size
    # numBlocks = 2 
    # numClients = 1000

    
    if rank >= 3:
        for i in range(0, numClients):
            sentStart = modifiedCommunicator.totalBitsSent
            sentStart0 = modifiedCommunicator.numberBitsSent0
            sentStart1 = modifiedCommunicator.numberBitsSent1
            sentStart2 = modifiedCommunicator.numberBitsSent2
            recievedStart = modifiedCommunicator.totalBitsRecieved
            recievedStart0 = modifiedCommunicator.numberBitsRecieved0
            recievedStart1 = modifiedCommunicator.numberBitsRecieved1
            recievedStart2 = modifiedCommunicator.numberBitsRecieved2
            timeStart = time.time()
            
            client.clientSimulation(rank, numBlocks=numBlocks)
            
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

        print("logs/ab/client.txt", "a")
        print("\n--------------------------------------\n--------------------------------------\n")
        print(f"NumClients: {numClients}\n")
        print(f"MsgSize: {numBlocks*16}\n")
        print("---------------online------------------\n")
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
       

    elif rank == 0:
        key1 = []
        tagShare1 = []
        ek1 = []
        ctshares1 = []

        #Get size of table
        n = numClients
        for i in range(0, numClients):
            te = modifiedCommunicator.myRecieve(3)
            key1.append(te)
            te = modifiedCommunicator.myRecieve(3)
            tagShare1.append(te)
            teArr = []
            for j in range(0, numBlocks):
                te = modifiedCommunicator.myRecieve(3)
                teArr.append(te)
            ctshares1.append(teArr)
            te = modifiedCommunicator.myRecieve(3)
            ek1.append(te)


        #Receive mult triples
        server.receiveTriples((numBlocks + 1)*n)
        server.receiveTriplesSecond(n)

        shareTable = []
        #Broadcast mask of shares
        sentStart = modifiedCommunicator.totalBitsSent
        sentStart0 = modifiedCommunicator.numberBitsSent0
        sentStart1 = modifiedCommunicator.numberBitsSent1
        sentStart2 = modifiedCommunicator.numberBitsSent2
        recievedStart = modifiedCommunicator.totalBitsRecieved
        recievedStart0 = modifiedCommunicator.numberBitsRecieved0
        recievedStart1 = modifiedCommunicator.numberBitsRecieved1
        recievedStart2 = modifiedCommunicator.numberBitsRecieved2
        
        timeStart = time.time()

        for i in range(0, numClients):
            expandedKey = server.broadcastShares(rank, key1[i], tagShare1[i], ctshares1[i], ek1[i], numBlocks + 1, (i)*(numBlocks + 1))
            arrRes = []
            for j in range(0, len(expandedKey)):
                arrRes.append(expandedKey[j])
            arrRes.append(tagShare1[i])
            for j in range(0, len(ctshares1[i])):
                arrRes.append(ctshares1[i][j])
            arrRes.append(ek1[i])
            shareTable.append(arrRes)

        #Permute using common share with rank 1 party 
        permutation = server.GenPerm(n, crypten.generators['permCommon'])
        [permutation1, a2Table, b2Table] = server.sampleA2B2(numBlocks, n)
        
        permutedTable = []


        for i in range(0, n):
            permutedTable.append([])

        for i in range(0, n):
            permutedTable[i] = shareTable[permutation[i]]
        
        

        s1 = server.performPermutation(rank, None, a2Table, b2Table, None, permutedTable, permutation1, n, 2*numBlocks + 3)

        #Send cts
        for i in range(0, n):
            for j in range(0, numBlocks):
                recValTe = modifiedCommunicator.mySend(s1[i][j + numBlocks + 2], 1)

        #Receive cts
        recVal = []
        for i in range(0, n):
            arr = []
            for j in range(0, numBlocks):
                recValTe = modifiedCommunicator.myRecieve(1)
                arr.append(recValTe)
               
            recVal.append(arr) 

        for i in range(0, n):
            for j in range(0, numBlocks):
                s1[i][j + numBlocks + 2] = modp.addp(recVal[i][j],  s1[i][j + numBlocks + 2])
        

        expectedTagSum = server.addTagsSecondMAC(rank, s1, numBlocks) 
        server.verifyTag(rank, expectedTagSum)

        hashedTable = server.getHashedTable(s1)
        recHash = modifiedCommunicator.myRecieve(1)
        modifiedCommunicator.mySend(hashedTable, 1)

        #Recieve and send share
        server.sendTable(s1, n, 2*numBlocks + 3, 1)
        s2 = server.recTable(n, 2*numBlocks + 3, 1)

        assert(server.getHashedTable(s2) == recHash)

        valTable = server.computeFromShares(s1, s2, numBlocks)
        server.checkMACSOutput(valTable, numBlocks)

        server.decryptMessages(valTable, numBlocks)

        timeEnd = time.time()
        totime = (timeEnd - timeStart)/32
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

        print("logs/ab/party0.txt", "a")
        print("\n--------------------------------------\n--------------------------------------\n")
        print(f"NumClients: {numClients}\n")
        print(f"MsgSize: {numBlocks*16}\n")
        print("---------------online------------------\n")
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



    elif rank == 1:
        key2 = []
        tagShare2 = []
        ek2 = []
        ctshares2 = []

        #Get size of table
        n = numClients

        for i in range(0, numClients):
            te = modifiedCommunicator.myRecieve(3)
            key2.append(te)
            te = modifiedCommunicator.myRecieve(3)
            tagShare2.append(te)
            teArr = []
            for j in range(0, numBlocks):
                te = modifiedCommunicator.myRecieve(3)
                teArr.append(te)
            ctshares2.append(teArr)
            te = modifiedCommunicator.myRecieve(3)
            ek2.append(te)

        #Receive mult triples
        server.receiveTriples((numBlocks + 1)*n)
        server.receiveTriplesSecond(n)

        #Broadcast mask of shares
        sentStart = modifiedCommunicator.totalBitsSent
        sentStart0 = modifiedCommunicator.numberBitsSent0
        sentStart1 = modifiedCommunicator.numberBitsSent1
        sentStart2 = modifiedCommunicator.numberBitsSent2
        recievedStart = modifiedCommunicator.totalBitsRecieved
        recievedStart0 = modifiedCommunicator.numberBitsRecieved0
        recievedStart1 = modifiedCommunicator.numberBitsRecieved1
        recievedStart2 = modifiedCommunicator.numberBitsRecieved2
        timeStart = time.time()

        shareTable = []
        for i in range(0, numClients):
            expandedKey = server.broadcastShares(rank, key2[i], tagShare2[i], ctshares2[i], ek2[i], numBlocks + 1, (i)*(numBlocks + 1))
            arrRes = []
            for j in range(0, len(expandedKey)):
                arrRes.append(expandedKey[j])
            arrRes.append(tagShare2[i])
            for j in range(0, len(ctshares2[i])):
                arrRes.append(ctshares2[i][j])
            arrRes.append(ek2[i])
            shareTable.append(arrRes)

        #Permute using common share with rank 0 party
        permutation = server.GenPerm(n, crypten.generators['permCommon'])

        #Sample random a1
        [permutation2, a1] = server.sampleA1(numBlocks, n)

        #Received Table
        deltaTable = server.recTable(n, 2*numBlocks + 3, 2)
        
        permutedTable = []
        for i in range(0, n):
            permutedTable.append([])

        for i in range(0, n):
            permutedTable[i] = shareTable[permutation[i]]
     
        s2 = server.performPermutation(rank, a1, None, None, deltaTable, permutedTable, permutation2, n, 2*numBlocks + 3)

        #Recieve all cts from l+3 to 2*l + 2
        recVal = []
        for i in range(0, n):
            arr = []
            for j in range(0, numBlocks):
                recValTe = modifiedCommunicator.myRecieve(0)
                arr.append(recValTe)
               
            recVal.append(arr)
        
        #Send cts
        for i in range(0, n):
            for j in range(0, numBlocks):
                recValTe = modifiedCommunicator.mySend(s2[i][j + numBlocks + 2], 0)

        for i in range(0, n):
            for j in range(0, numBlocks):
                s2[i][j + numBlocks + 2] = modp.addp(recVal[i][j],  s2[i][j + numBlocks + 2])

        expectedTagSum = server.addTagsSecondMAC(rank, s2, numBlocks) 
        server.verifyTag(rank, expectedTagSum)

        hashedTable = server.getHashedTable(s2)
        modifiedCommunicator.mySend(hashedTable, 0)
        recHash = modifiedCommunicator.myRecieve(0)

        #Recieve and send share
        s1 = server.recTable(n, 2*numBlocks + 3, 0)
        server.sendTable(s2, n, 2*numBlocks + 3, 0)

        assert(server.getHashedTable(s1) == recHash)
        valTable = server.computeFromShares(s1, s2, numBlocks)

        server.checkMACSOutput(valTable, numBlocks)

        server.decryptMessages(valTable, numBlocks)

        timeEnd = time.time()
        totime = (timeEnd - timeStart)/32
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

        print("logs/ab/party1.txt", "a")
        print("\n--------------------------------------\n--------------------------------------\n")
        print(f"NumClients: {numClients}\n")
        print(f"MsgSize: {numBlocks*16}\n")
        print("---------------online------------------\n")
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
    else:
        #Get size of table
        n = numClients
        #Generate l+1 multiplication triples
        server.generateTriplesAndCommunicate((numBlocks + 1)*n)
        server.generateTriplesAndCommunicate(n)

        #Receive 2 seeds
        seed1 = modifiedCommunicator.myRecieve(0)
        seed2 = modifiedCommunicator.myRecieve(0)

        seed3 = modifiedCommunicator.myRecieve(1)
        seed4 = modifiedCommunicator.myRecieve(1)

        sentStart = modifiedCommunicator.totalBitsSent
        sentStart0 = modifiedCommunicator.numberBitsSent0
        sentStart1 = modifiedCommunicator.numberBitsSent1
        sentStart2 = modifiedCommunicator.numberBitsSent2
        recievedStart = modifiedCommunicator.totalBitsRecieved
        recievedStart0 = modifiedCommunicator.numberBitsRecieved0
        recievedStart1 = modifiedCommunicator.numberBitsRecieved1
        recievedStart2 = modifiedCommunicator.numberBitsRecieved2
        
        timeStart = time.time()

        server.computeDelta(seed1, seed2, seed3, seed4, n, numBlocks)

        timeEnd = time.time()
        totime = (timeEnd - timeStart)/32
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

        print("logs/ab/party2.txt", "a")
        print("\n--------------------------------------\n--------------------------------------\n")
        print(f"NumClients: {numClients}\n")
        print(f"MsgSize: {numBlocks*16}\n")
        print("---------------online------------------\n")
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

clar(2, 10)