from ast import mod
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
import modifiedCommunicator
import hashlib
import math

#Output commitments
recieved_hash = []
#jmp send hashes
#Firs array to previous party, second array to next party
hashes_to_send_jmpsend = [[], []]
#jmp verify hashes
#First one recieved from previous party next one recieved from next party
values_to_verify = [[], []]
bitState = 0


def jmpSend(value1, value2, p1, p2, p3, rank):
        returnVal = 0
        if rank == p3:
            returnVal = np.uint64(modifiedCommunicator.myRecieve(p1))
            if p2 == (p3-1)%3:
                values_to_verify[0].append(returnVal)
            else:
                values_to_verify[1].append(returnVal)

        elif rank == p1:
            modifiedCommunicator.mySend(value1, p3)
            returnVal = value1
        else:
            if p3 == (p2-1)%3:
                hashes_to_send_jmpsend[0].append(value2)
            else:
                hashes_to_send_jmpsend[1].append(value2)

            returnVal = value2

        return returnVal

def getCommitmentHash(mask):
    hashed_string = hashlib.sha256(str(mask).encode('utf-8')).hexdigest()
    smallerString = hashed_string[0:10]
    hashed_string = int(smallerString, 16)%(2**64)
    return hashed_string

def getFullCommitmentHash(mask):
    hashed_string = hashlib.sha256(str(mask).encode('utf-8')).hexdigest()
    hash = [int(hashed_string[:16],16), int(hashed_string[16:32],16), int(hashed_string[32:48],16), int(hashed_string[48:64],16)]
    return hash

def circulateCommitments(rank, output):
    pos = output.preprocessingCount - 1
    print("here1", output.masks[pos])

    for i in range(0, len(output.masks[pos])):
        hashed_string1 = getCommitmentHash(output.masks[pos][i][0])
        hashed_string2 = getCommitmentHash(output.masks[pos][i][1])

        hash1 = jmpSend(hashed_string1, hashed_string2, 0, 2, 1, rank)
        hash2 = jmpSend(hashed_string2, hashed_string1, 0, 1, 2, rank)
        hash3 = jmpSend(hashed_string2, hashed_string1,1, 2, 0, rank)

        if rank == 0:
            recieved_hash.append(hash3)
        elif rank == 1:
            recieved_hash.append(hash1)
        else:
            recieved_hash.append(hash2)
        
def circulateCommitmentsBoolean(rank, output):
    pos = output.preProcessingCount - 1
    print("here", output.masks[pos])
    for i in range(0, len(output.masks[pos])):
        hashed_string1 = getCommitmentHash(output.masks[pos][i][0])
        hashed_string2 = getCommitmentHash(output.masks[pos][i][1])

        hash1 = jmpSend(hashed_string1, hashed_string2, 0, 2, 1, rank)
        hash2 = jmpSend(hashed_string2, hashed_string1, 0, 1, 2, rank)
        hash3 = jmpSend(hashed_string2, hashed_string1,1, 2, 0, rank)

        if rank == 0:
            recieved_hash.append(hash3)
        elif rank == 1:
            recieved_hash.append(hash1)
        else:
            recieved_hash.append(hash2)

def convertHexHashToInteger(hashOfMsgVal):
    hashAsInteger = int(hashOfMsgVal.hexdigest(), 16)
    hashAsInteger = hashAsInteger % (2 ** 64)

    return hashAsInteger

def sendHashToParty(fromIndex, toIndex):
    msgVal = ""
    for i in range(len(hashes_to_send_jmpsend[fromIndex])):
            msgVal = msgVal + str(hashes_to_send_jmpsend[fromIndex][i])
    hashOfMsgVal = hashlib.sha256(msgVal.encode())
    hashAsInteger = convertHexHashToInteger(hashOfMsgVal)
    modifiedCommunicator.mySend(hashAsInteger, toIndex)

def recieveAndVerify(toIndex, fromIndex, rank):
    msgVal = ""
    global bitState
    for i in range(len(values_to_verify[toIndex])):
            msgVal = msgVal + str(values_to_verify[toIndex][i])
    hashOfMsgVal = hashlib.sha256(msgVal.encode())
    hashAsInteger = convertHexHashToInteger(hashOfMsgVal)
    receievedVal = modifiedCommunicator.myRecieve(fromIndex)
    
    #Send and receive bits

    if receievedVal != hashAsInteger:
        if fromIndex == (rank - 1)%3:
            bitState = bitState + 1
        else:
            bitState = bitState + 2
    
    print(receievedVal, hashAsInteger)

def verifyJmpSend(rank):
    if rank == 0:
        recieveAndVerify(1, 1, rank)
        sendHashToParty(0, 2)
        sendHashToParty(1, 1)
        recieveAndVerify(0, 2, rank)
    elif rank == 1:
        #First send combined hash
        sendHashToParty(0, 0)
        sendHashToParty(1, 2)
        recieveAndVerify(0, 0, rank)
        recieveAndVerify(1, 2, rank)
    else:
        recieveAndVerify(0, 1, rank)
        recieveAndVerify(1, 0, rank)
        sendHashToParty(1, 0)
        sendHashToParty(0, 1)

    #Send around validness bits
    if rank == 0:
        bitState1 = modifiedCommunicator.myRecieve(1)
        modifiedCommunicator.mySend(bitState, 2)
        modifiedCommunicator.mySend(bitState, 1)
        bitState2 = modifiedCommunicator.myRecieve(2)
    elif rank == 1:
        #First send combined hash
        modifiedCommunicator.mySend(bitState, 0)
        modifiedCommunicator.mySend(bitState, 2)
        bitState1 = modifiedCommunicator.myRecieve(0)
        bitState2 = modifiedCommunicator.myRecieve(2)
    else:
        bitState1 = modifiedCommunicator.myRecieve(1)
        bitState2 = modifiedCommunicator.myRecieve(0)
        modifiedCommunicator.mySend(bitState, 0)
        modifiedCommunicator.mySend(bitState, 1)

def openCommitments(rank, output):
    pos = output.onlineCount - 1
    for i in range(0, len(output.share)):
        sum = np.uint64(0)
        val1 = 0
        val2 = 0
        if rank == 0:
            #Send to party 1 and 2
            modifiedCommunicator.mySend(output.masks[pos][i][0], 1)
            modifiedCommunicator.mySend(output.masks[pos][i][1], 2)

            #Recieve from party 1
            val2 = np.uint64(modifiedCommunicator.myRecieve(1))
            val1 = np.uint64(modifiedCommunicator.myRecieve(2))
            sum = np.uint64(val1 + output.masks[pos][i][0] + output.masks[pos][i][1])

        elif rank == 1:
            #Recieve from party 0
            val1 = modifiedCommunicator.myRecieve(0)
            val2 = modifiedCommunicator.myRecieve(2)
            #send to party 0
            modifiedCommunicator.mySend(output.masks[pos][i][1], 0)
            modifiedCommunicator.mySend(output.masks[pos][i][0], 2)
            sum = np.uint64(val1 + output.masks[pos][i][0] + output.masks[pos][i][1])

        else:
        
            #Recieve from party 0
            val2 = modifiedCommunicator.myRecieve(0)
            modifiedCommunicator.mySend(output.masks[pos][i][1], 1)
            modifiedCommunicator.mySend(output.masks[pos][i][0], 0)
            val1 = modifiedCommunicator.myRecieve(1)
            sum = np.uint64(val1 + output.masks[pos][i][0] + output.masks[pos][i][1])

        hash1 = getCommitmentHash(val1)
        hash2 = getCommitmentHash(val2)
        output.share[i] = np.uint64(output.share[i] - sum)
       
    output.decodeShares()

    return output

def performBitwiseXOR(vala, valb, valc):
    valres = 0
    cur = 1
    for i in range(0, 64):
        bita = int(vala%2)
        bitb = int(valb%2)
        bitc = int(valc%2)
        valres = valres + cur*(bita^bitb^bitc)
        cur = cur*2
        vala = math.floor(vala/2)
        valb = math.floor(valb/2)
        valc = math.floor(valc/2)

    return valres

def performBitwiseXOR2(vala, valb):
    valres = 0
    cur = 1
    for i in range(0, 64):
        bita = int(vala%2)
        bitb = int(valb%2)
    
        valres = valres + cur*(bita^bitb)
        cur = cur*2
        vala = math.floor(vala/2)
        valb = math.floor(valb/2)
    

    return valres

def openCommitmentsBoolean(rank, output):
    pos = output.onlineCount - 1
    for i in range(0, len(output.share)):
        sum = np.uint64(0)
        val1 = 0
        val2 = 0
        if rank == 0:
            #Send to party 1 and 2
            modifiedCommunicator.mySend(output.masks[pos][i][0], 1)
            modifiedCommunicator.mySend(output.masks[pos][i][1], 2)

            #Recieve from party 1
            val2 = np.uint64(modifiedCommunicator.myRecieve(1))
            val1 = np.uint64(modifiedCommunicator.myRecieve(2))
            sum1 = val1^output.masks[pos][i][0]
            sum = sum1^output.masks[pos][i][1]

        elif rank == 1:
            #Recieve from party 0
            val1 = modifiedCommunicator.myRecieve(0)
            val2 = modifiedCommunicator.myRecieve(2)
            #send to party 0
            modifiedCommunicator.mySend(output.masks[pos][i][1], 0)
            modifiedCommunicator.mySend(output.masks[pos][i][0], 2)
            sum1 = val1^output.masks[pos][i][0]
            sum = sum1^output.masks[pos][i][1]
        else:
        
            #Recieve from party 0
            val2 = modifiedCommunicator.myRecieve(0)
            modifiedCommunicator.mySend(output.masks[pos][i][1], 1)
            modifiedCommunicator.mySend(output.masks[pos][i][0], 0)
            val1 = modifiedCommunicator.myRecieve(1)
            sum1 = val1^output.masks[pos][i][0]
            sum = sum1^output.masks[pos][i][1]
        hash1 = getCommitmentHash(val1)
        hash2 = getCommitmentHash(val2)
        output.share[i] = output.share[i]^sum
        
    if rank == 0:
        print(output.share)
   

    return output