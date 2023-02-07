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
def cshuffle(numBlocks, numClients):
    
    rank = comm.get().rank
    world_size = comm.get().world_size
    # numBlocks = 2 
    # numClients = 1000

    MtimeS = time.time()
    # if rank >= 3:
    #     for i in range(0, numClients):
    #         client.clientSimulation(rank, numBlocks=numBlocks)

    if rank == 0:
        n = numClients

        #Receive mult triples
        server.receiveTriples((numBlocks + 1)*n)
        server.receiveTriplesSecond(n)

        shareTable =[[0]*(2*numBlocks + 3) for i in range(0, n)]

       
        #Permute using common share with rank 1 party 
        permutation = [0 for i in range(0, n)]
        a2Table = [[0]*(2*numBlocks + 3) for i in range(0, n)]
        b2Table = [[0]*(2*numBlocks + 3) for i in range(0, n)]

        permutation1 = [0 for i in range(0, n)]
        # [permutation1, a2Table, b2Table] = server.sampleA2B2(numBlocks, n)
        
        permutedTable = []

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
        totime =0
        timeStart = time.time()

        permutedTable = [[0]*(2*numBlocks + 3) for i in range(0, n)]

        timeEnd = time.time()
        totime += (timeEnd - timeStart)/32
        timeStart = time.time()
        s1 = server.performPermutation(rank, None, a2Table, b2Table, None, permutedTable, permutation1, n, 2*numBlocks + 3)
        
        timeEnd = time.time()
        totime += timeEnd - timeStart
        timeStart = time.time()

        #Send cts
        valsToBeSent = []
        for i in range(0, n):
            cur = []
            for j in range(0, numBlocks):
                cur.append(s1[i][j + numBlocks + 2])
                # recValTe = modifiedCommunicator.mySend(s1[i][j + numBlocks + 2], 1)
            valsToBeSent.append(cur)

        modifiedCommunicator.mySend1(valsToBeSent, 1)

        #Receive cts
        recVal = modifiedCommunicator.myRecieve1(1, [n, numBlocks, 2])
        # for i in range(0, n):
        #     arr = []
        #     for j in range(0, numBlocks):
        #         recValTe = modifiedCommunicator.myRecieve(1)
        #         arr.append(recValTe)
               
        #     recVal.append(arr) 

        for i in range(0, n):
            for j in range(0, numBlocks):
                s1[i][j + numBlocks + 2] = modp.addp(recVal[i][j],  s1[i][j + numBlocks + 2])

        expectedTagSum = server.addTagsSecondMAC(rank, s1, numBlocks) 
        server.verifyTag(rank, expectedTagSum)

        timeEnd = time.time()
        print(rank, timeEnd - timeStart)
        totime += (timeEnd - timeStart)/32
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

        f = open("logs/shuffle/party0.txt", "a")
        f.write("\n--------------------------------------\n--------------------------------------\n")
        f.write(f"NumClients: {numClients}\n")
        f.write(f"MsgSize: {numBlocks*32}\n")
        f.write("---------------online------------------\n")
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

        # hashedTable = server.getHashedTable(s1)
        # recHash = modifiedCommunicator.myRecieve(1)
        # modifiedCommunicator.mySend(hashedTable, 1)

        # #Recieve and send share
        # server.sendTable(s1, n, 2*numBlocks + 3, 1)
        # s2 = server.recTable(n, 2*numBlocks + 3, 1)

        # assert(server.getHashedTable(s2) == recHash)

        # valTable = server.computeFromShares(s1, s2, numBlocks)
        # server.checkMACSOutput(valTable, numBlocks)

        # server.decryptMessages(valTable, numBlocks)
    elif rank == 1:
        n = numClients
        #Receive mult triples
        server.receiveTriples((numBlocks + 1)*n)
        server.receiveTriplesSecond(n)

        #Broadcast mask of shares
        shareTable = []

        shareTable =[[0]*(2*numBlocks + 3) for i in range(0, n)]


        #Permute using common share with rank 0 party
        permutation = [0 for i in range(0, n)]

        #Sample random a1
        a1 = [[0]*(2*numBlocks + 3) for i in range(0, n)]
        permutation2 = [0 for i in range(0, n)]
        # [permutation2, a1] = server.sampleA1(numBlocks, n)

        #Received Table
        deltaTable =[[0]*(2*numBlocks + 3) for i in range(0, n)]
        # deltaTable = server.recTable(n, 2*numBlocks + 3, 2)

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
        totime =0
        timeStart = time.time()
        
        permutedTable = [[0]*(2*numBlocks + 3) for i in range(0, n)]

        timeEnd = time.time()
        totime += (timeEnd - timeStart)/32
        timeStart = time.time()

        s2 = server.performPermutation(rank, a1, None, None, deltaTable, permutedTable, permutation2, n, 2*numBlocks + 3)

        timeEnd = time.time()
        totime += timeEnd - timeStart
        timeStart = time.time()


        #Receive cts
        recVal = modifiedCommunicator.myRecieve1(0, [n, numBlocks, 2])
        
        #Send cts
        valsToBeSent = []
        for i in range(0, n):
            cur = []
            for j in range(0, numBlocks):
                cur.append(s2[i][j + numBlocks + 2])
                # recValTe = modifiedCommunicator.mySend(s1[i][j + numBlocks + 2], 1)
            valsToBeSent.append(cur)
        modifiedCommunicator.mySend1(valsToBeSent, 0)
       
        #Recieve all cts from l+3 to 2*l + 2
        # recVal = []
        # for i in range(0, n):
        #     arr = []
        #     for j in range(0, numBlocks):
        #         recValTe = modifiedCommunicator.myRecieve(0)
        #         arr.append(recValTe)
               
        #     recVal.append(arr)
        
        # #Send cts
        # for i in range(0, n):
        #     for j in range(0, numBlocks):
        #         recValTe = modifiedCommunicator.mySend(s2[i][j + numBlocks + 2], 0)

        for i in range(0, n):
            for j in range(0, numBlocks):
                s2[i][j + numBlocks + 2] = modp.addp(recVal[i][j],  s2[i][j + numBlocks + 2])

        expectedTagSum = server.addTagsSecondMAC(rank, s2, numBlocks) 
        server.verifyTag(rank, expectedTagSum)

        timeEnd = time.time()
        totime += (timeEnd - timeStart)/32
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

        f = open("logs/shuffle/party1.txt", "a")
        f.write("\n--------------------------------------\n--------------------------------------\n")
        f.write(f"NumClients: {numClients}\n")
        f.write(f"MsgSize: {numBlocks*32}\n")
        f.write("---------------online------------------\n")
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


        # hashedTable = server.getHashedTable(s2)
        # modifiedCommunicator.mySend(hashedTable, 0)
        # recHash = modifiedCommunicator.myRecieve(0)

        # #Recieve and send share
        # s1 = server.recTable(n, 2*numBlocks + 3, 0)
        # server.sendTable(s2, n, 2*numBlocks + 3, 0)

        # assert(server.getHashedTable(s1) == recHash)
        # valTable = server.computeFromShares(s1, s2, numBlocks)

        # server.checkMACSOutput(valTable, numBlocks)

        # server.decryptMessages(valTable, numBlocks)
    else:
        #Get size of table
        n = numClients
        #Generate l+1 multiplication triples
        server.generateTriplesAndCommunicate((numBlocks + 1)*n)
        server.generateTriplesAndCommunicate(n)

        #Receive 2 seeds
        # seed1 = modifiedCommunicator.myRecieve(0)
        # seed2 = modifiedCommunicator.myRecieve(0)

        # seed3 = modifiedCommunicator.myRecieve(1)
        # seed4 = modifiedCommunicator.myRecieve(1)

        sentStart = modifiedCommunicator.totalBitsSent
        sentStart0 = modifiedCommunicator.numberBitsSent0
        sentStart1 = modifiedCommunicator.numberBitsSent1
        sentStart2 = modifiedCommunicator.numberBitsSent2
        recievedStart = modifiedCommunicator.totalBitsRecieved
        recievedStart0 = modifiedCommunicator.numberBitsRecieved0
        recievedStart1 = modifiedCommunicator.numberBitsRecieved1
        recievedStart2 = modifiedCommunicator.numberBitsRecieved2
        timeStart = time.time()

        # server.computeDelta(seed1, seed2, seed3, seed4, n, numBlocks)

        timeEnd = time.time()
        totime =  timeEnd - timeStart
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

        f = open("logs/shuffle/party2.txt", "a")
        f.write("\n--------------------------------------\n--------------------------------------\n")
        f.write(f"NumClients: {numClients}\n")
        f.write(f"MsgSize: {numBlocks*32}\n")
        f.write("---------------online------------------\n")
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
    f.close()


