from ast import mod
from audioop import mul
from glob import glob
import sys

from sklearn import cluster


sys.path.insert(0, '../../')

from ctypes import addressof
import cryptenLocal as crypten
import cryptenLocal.clarion as clarion
import cryptenLocal.communicator as comm
import cryptenLocal.clarion.modifiedCommunicator as modifiedCommunicator
import cryptenLocal.SwiftMPC.primitives.arithmetic as arithmetic
import torch
import sys
import os
import hashlib
import time
import warnings
import numpy as np
import binascii
import modp
from Crypto.Cipher import AES
from Crypto.Util import Counter
import client
import hashlib


a = []
b = []
c = []
asec = []
bsec = []
csec = []

#generate a permutation of the numbers [0, n)
def GenPerm(n , seed):
    perm = [0 for i in range(0, n)]
    randomness = client.AesPRG(4*n, seed)
    
    for i in range(1, n):
        j = int.from_bytes(randomness[4*i:4*(i+1)], byteorder = 'big') % (i+1)
        perm[i] = perm[j]
        perm[j] = i
    
    return perm


def generateTriplesAndCommunicate(numTriples):
    aSeed = os.urandom(16)
    bSeed = os.urandom(16)

    aExpanded = client.AesPRG(numTriples, aSeed)
    bExpanded = client.AesPRG(numTriples, bSeed)

    aAsInt = []
    bAsInt = []

    for i in range(0, numTriples):
        vala = int.from_bytes(aExpanded[16*i:16*i + 16], byteorder = 'big')
        valb = int.from_bytes(bExpanded[16*i:16*i + 16], byteorder = 'big')
        valc = modp.multp(vala, valb)

        [sharea1, sharea2] = modp.getAdditiveSecretShares(vala)
        modifiedCommunicator.mySend(sharea1, 0)
        modifiedCommunicator.mySend(sharea2, 1)

        [shareb1, shareb2] = modp.getAdditiveSecretShares(valb)
        modifiedCommunicator.mySend(shareb1, 0)
        modifiedCommunicator.mySend(shareb2, 1)

        [sharec1, sharec2] = modp.getAdditiveSecretShares(valc)
        modifiedCommunicator.mySend(sharec1, 0)
        modifiedCommunicator.mySend(sharec2, 1)


def receiveTriples(numTriples):

    global a, b, c

    for i in range(0, numTriples):
        rec = modifiedCommunicator.myRecieve(2)
        a.append(rec)

        rec = modifiedCommunicator.myRecieve(2)
        b.append(rec)

        rec = modifiedCommunicator.myRecieve(2)
        c.append(rec)

def receiveTriplesSecond(numTriples):

    global asec, bsec, csec

    for i in range(0, numTriples):
        rec = modifiedCommunicator.myRecieve(2)
        asec.append(rec)

        rec = modifiedCommunicator.myRecieve(2)
        bsec.append(rec)

        rec = modifiedCommunicator.myRecieve(2)
        csec.append(rec)


def broadcastShares(rank, key, tagShare, ctshares, ek, numBlocks, btpos):

    expandedKey = client.AesPRG(numBlocks, key.to_bytes(16, byteorder = 'big'))
    
    mulShare = 0
    global a, b, c
    
    for i in range(0, numBlocks):
        if i != numBlocks - 1:
            valKey = int.from_bytes(expandedKey[16*i:16*i+16], byteorder = 'big')
            valCt = ctshares[i]
            valKeyMasked = modp.addp(valKey, -a[btpos + i])
            valCtMasked = modp.addp(valCt, -b[btpos + i])
            valKeyMaskedOther = 0
            valCtMaskedOther = 0
            #Recieve and Send masked shares
            if rank == 0:
                valKeyMaskedOther = modifiedCommunicator.myRecieve(1)
                valCtMaskedOther = modifiedCommunicator.myRecieve(1)

                modifiedCommunicator.mySend(valKeyMasked, 1)
                modifiedCommunicator.mySend(valCtMasked, 1)
            else:
                modifiedCommunicator.mySend(valKeyMasked, 0)
                modifiedCommunicator.mySend(valCtMasked, 0)

                valKeyMaskedOther = modifiedCommunicator.myRecieve(0)
                valCtMaskedOther = modifiedCommunicator.myRecieve(0)

            val1ct = modp.addp(valKeyMasked, valKeyMaskedOther)
            val2ct = modp.addp(valCtMasked, valCtMaskedOther)
            prod = modp.multp(val1ct, val2ct)
            if rank == 0:
                prod = 0
            valRes = modp.addp(prod, c[btpos + i])

            prod1 = modp.multp(val1ct, b[btpos + i])
            prod2 = modp.multp(val2ct, a[btpos + i])

            valRes = modp.addp(valRes, prod1)
            valRes = modp.addp(valRes, prod2)

            mulShare = modp.addp(mulShare, valRes) 

        else:
            valKey = int.from_bytes(expandedKey[16*i:16*i+16], byteorder = 'big')
            valCt = ek
            valKeyMasked = modp.addp(valKey, -a[btpos + i])
            valCtMasked = modp.addp(valCt, -b[btpos + i])
            valKeyMaskedOther = 0
            valCtMaskedOther = 0

            #Recieve and Send masked shares
            if rank == 0:
                valKeyMaskedOther = modifiedCommunicator.myRecieve(1)
                valCtMaskedOther = modifiedCommunicator.myRecieve(1)

                modifiedCommunicator.mySend(valKeyMasked, 1)
                modifiedCommunicator.mySend(valCtMasked, 1)
            else:
                modifiedCommunicator.mySend(valKeyMasked, 0)
                modifiedCommunicator.mySend(valCtMasked, 0)

                valKeyMaskedOther = modifiedCommunicator.myRecieve(0)
                valCtMaskedOther = modifiedCommunicator.myRecieve(0)

            val1ct = modp.addp(valKeyMasked, valKeyMaskedOther)
            val2ct = modp.addp(valCtMasked, valCtMaskedOther)
            prod = modp.multp(val1ct, val2ct)
            if rank == 0:
                prod = 0
            valRes = modp.addp(prod, c[btpos + i])

            prod1 = modp.multp(val1ct, b[btpos + i])
            prod2 = modp.multp(val2ct, a[btpos + i])

            valRes = modp.addp(valRes, prod1)
            valRes = modp.addp(valRes, prod2)

            mulShare = modp.addp(mulShare, valRes) 

    diffShare = modp.addp(tagShare, -mulShare)
    diffShareOther = 0
    if rank == 0:
        diffShareOther = modifiedCommunicator.myRecieve(1)
        modifiedCommunicator.mySend(diffShare, 1)
    else:
        modifiedCommunicator.mySend(diffShare, 0)
        diffShareOther = modifiedCommunicator.myRecieve(0)

    assert(modp.addp(diffShare, diffShareOther) == 0)
    
    expandedKeyInt = []

    for i in range(0, numBlocks):
        expandedKeyInt.append(int.from_bytes(expandedKey[16*i:16*i + 16], byteorder = 'big'))

    return expandedKeyInt
    
def sampleA2B2(numBlocks, n):
    randomSeed = os.urandom(16)
    randomSeed1 = os.urandom(16)
    permutation1 = GenPerm(n, randomSeed)

    a2b2 = client.AesPRG(2*n*(2*numBlocks + 3), randomSeed1)

    a2 = a2b2[0:n*16*(2*numBlocks + 3)]
    b2 = a2b2[n*16*(2*numBlocks + 3):]

    a2Table = convertByteStringToTable(n, 2*numBlocks + 3, a2)
    b2Table = convertByteStringToTable(n, 2*numBlocks + 3, b2)

    # #Sends seed to P3
    modifiedCommunicator.mySend(randomSeed, 2)
    modifiedCommunicator.mySend(randomSeed1, 2)

    return [permutation1, a2Table, b2Table]

    
def sampleA1(numBlocks, n):
    randomSeed = os.urandom(16)
    randomSeed1 = os.urandom(16)

    permutation2 = GenPerm(n, randomSeed)
    a1 = client.AesPRG(n*(2*numBlocks + 3), randomSeed1)

    a1Table = convertByteStringToTable(n, 2*numBlocks + 3, a1)
    #Sends seed to P3
    modifiedCommunicator.mySend(randomSeed, 2)
    modifiedCommunicator.mySend(randomSeed1, 2)

    return [permutation2, a1Table]


def permuteTable(table, permutation):

    resPerm = []
    for i in range(0, len(permutation)):
        resPerm.append(table[permutation[i]].copy())
    return resPerm

def addToTable(table1, table2):
    
    resTable = []
    for i in range(0, len(table1)):
        arr = []
        for j in range(0, len(table1[0])):
            arr.append(modp.addp(table1[i][j], table2[i][j]))
        resTable.append(arr)

    return resTable

def subFromTable(table1, table2):

    resTable = []
    for i in range(0, len(table1)):
        arr = []
        for j in range(0, len(table1[0])):
            arr.append(modp.addp(table1[i][j], -table2[i][j]))
        resTable.append(arr)
    
    return resTable


def sendTable(table, n, m, dst):
    for i in range(0, n):
        for j in range(0, m):
            modifiedCommunicator.mySend(table[i][j], dst)

def recTable(n, m, src):
    resTable = []

    for i in range(0, n):
        arr = []
        for j in range(0, m):
            recValInt = modifiedCommunicator.myRecieve(src)
            arr.append(recValInt)
        resTable.append(arr)
    
    return resTable

def convertByteTableToIntTable(table, n, m):
    resTable = []

    for i in range(0, n):
        curArr = []
        for j in range(0, m):
            curArr.append(int.from_bytes(table[i][16*j:16*j+16], byteorder = "big"))

        resTable.append(curArr)

    return resTable

def convertByteStringToTable(n, m, str):
    table = []
    for i in range(0, n):
        arr = []
        for j in range(0, m):
            arr.append(int.from_bytes(str[i*m*16 + j*16:i*m*16 + j*16 + 16], byteorder = 'big'))
        table.append(arr)
    return table

def computeDelta(seed1, seed2, seed3, seed4, n, numBlocks):

    #Convert seeds to bytes
    seed1 = seed1.to_bytes(16, byteorder = "big")
    seed2 = seed2.to_bytes(16, byteorder = "big")
    seed3 = seed3.to_bytes(16, byteorder = "big")
    seed4 = seed4.to_bytes(16, byteorder = "big")

    permutation1 = GenPerm(n, seed1)
    a2b2 = client.AesPRG(2*n*(2*numBlocks + 3), seed2)
    permutation2 = GenPerm(n, seed3)
    # print("perm1", permutation1)
    # print("perm2", permutation2)
    a1 = client.AesPRG(n*(2*numBlocks + 3), seed4)
   
    a2 = a2b2[0:n*16*(2*numBlocks + 3)]
    b2 = a2b2[n*16*(2*numBlocks + 3):]
    
    a2Table = convertByteStringToTable(n, 2*numBlocks + 3, a2)
    b2Table = convertByteStringToTable(n, 2*numBlocks + 3, b2)

    a1Table = convertByteStringToTable(n, 2*numBlocks + 3, a1)


    #Permute a1 table
    perma1 = permuteTable(a1Table, permutation1)

    #Add a2 to the permutation
    addedTable = addToTable(perma1, a2Table)

    #Permute using p2
    perma2 = permuteTable(addedTable, permutation2)
    #Subtract b2
    subTable = subFromTable(perma2, b2Table)

    sendTable(subTable, n, 2*numBlocks + 3, 1)

def beaversMultiply(val1, val2, index, rank):
    global asec, bsec, csec
    val1Masked = modp.addp(val1, -asec[index])
    val2Masked = modp.addp(val2, -bsec[index])

    if rank == 0:
        val1MaskedOther = modifiedCommunicator.myRecieve(1)
        val2MaskedOther = modifiedCommunicator.myRecieve(1)

        modifiedCommunicator.mySend(val1Masked, 1)
        modifiedCommunicator.mySend(val2Masked, 1)
    else:
        modifiedCommunicator.mySend(val1Masked, 0)
        modifiedCommunicator.mySend(val2Masked, 0)

        val1MaskedOther = modifiedCommunicator.myRecieve(0)
        val2MaskedOther = modifiedCommunicator.myRecieve(0)

    val1Masked = modp.addp(val1Masked, val1MaskedOther)
    val2Masked = modp.addp(val2Masked, val2MaskedOther)
    prod = modp.multp(val1Masked, val2Masked)
    if rank == 0:
        prod = 0
    share = modp.addp(csec[index], modp.addp(prod, modp.addp(modp.multp(val1Masked, bsec[index]), modp.multp(val2Masked, asec[index]))))
    return share

def addTagsSecondMAC(rank, share, numBlocks):

    res = 0
    res1 = 0
    n = len(share)
    m = len(share[0])
    for i in range(0, n):
        for j in range(0, numBlocks):
            res = modp.addp(res, modp.multp(share[i][j], share[i][j + numBlocks + 2]))

    for i in range(0, n):
        val = beaversMultiply(share[i][numBlocks], share[i][m - 1], i, rank)
        res = modp.addp(res, val)

    for i in range(0, n):
        res1 = modp.addp(res1, share[i][numBlocks + 1])

    res = modp.addp(res1, -res)

    return res

def performPermutation(rank, a1Table, a2Table, b2Table, deltaTable, shareTable, permutation, n, m):
    if rank == 0:
        z2 = recTable(n, m, 1)
        z1 = subFromTable(permuteTable(addToTable(z2, shareTable), permutation), a2Table)
        sendTable(z1, n, m, 1)
        s1 = b2Table
     
        return s1
    else:
        z2 = subFromTable(shareTable, a1Table)
        sendTable(z2, n, m, 0)
        z1 = recTable( n, m, 0)
        s2 = addToTable(permuteTable(z1, permutation), deltaTable)
       
        return s2

def hashVal(val):
    hashTag = hashlib.sha256(val.to_bytes(16, byteorder = 'big')).hexdigest()
    sendVal = int(hashTag[0:16], 16)
    return sendVal

def verifyTag(rank, expectedTagSum):
    if rank == 0:
        recHash = modifiedCommunicator.myRecieve(1)
       
        modifiedCommunicator.mySend(hashVal(expectedTagSum), 1)

        #Send and receive share
        recVal = modifiedCommunicator.myRecieve(1)
        modifiedCommunicator.mySend(expectedTagSum, 1)

        assert(hashVal(recVal) == recHash)
        assert(modp.addp(recVal, expectedTagSum) == 0)
    else:
       
        modifiedCommunicator.mySend(hashVal(expectedTagSum), 0)
        recHash = modifiedCommunicator.myRecieve(0)

        #Send receive share
        modifiedCommunicator.mySend(expectedTagSum, 0)
        recVal = modifiedCommunicator.myRecieve(0)

        assert(hashVal(recVal) == recHash)
        assert(modp.addp(recVal, expectedTagSum) == 0)


def getHashedTable(table):
    byteString = b""

    for i in range(0, len(table)):
        for j in range(0, len(table[0])):
            byteString = byteString + table[i][j].to_bytes(16, byteorder = "big")

    hashTag = hashlib.sha256(byteString).hexdigest()
    sendVal = int(hashTag[0:16], 16) 

    return sendVal

def computeFromShares(s1, s2, numBlocks):

    for i in range(0, len(s1)):
        for j in range(0, len(s1[0])):
            if j <= numBlocks + 1 or j == len(s1[0]) - 1:
                s1[i][j] = modp.addp(s1[i][j], s2[i][j])

    return s1


def checkMACSOutput(valTable, numBlocks):
    n = len(valTable)
    m = len(valTable[0])
    for i in range(0, n):
        curVal = 0
        for j in range(0, numBlocks):
            curVal = modp.addp(curVal, modp.multp(valTable[i][j], valTable[i][j + numBlocks + 2]))
         

        curVal = modp.addp(curVal, modp.multp(valTable[i][numBlocks], valTable[i][m-1]))
        assert(curVal == valTable[i][numBlocks + 1])

def decryptMessages(valTable, numBlocks): 
    n = len(valTable)
    m = len(valTable[0])

    for i in range(0, n):
        cipherText = b""
        for j in range(numBlocks + 2, m - 1):
            cipherText = cipherText + valTable[i][j].to_bytes(16, byteorder = "big")
        
        decryptedMessage = client.decrypt_message(valTable[i][m-1].to_bytes(16, byteorder = "big"), cipherText)
        # print(decryptedMessage)
