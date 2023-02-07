from ast import mod
import sys


sys.path.insert(0, '../../')

from ctypes import addressof
import cryptenLocal as crypten
import cryptenLocal.SwiftMPC as swift
import cryptenLocal.communicator as comm
import cryptenLocal.SwiftMPC.primitives.boolean as boolean
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
from cryptenLocal.common.rng import generate_random_shareFromAES
import modifiedCommunicator

def getRandomPermutation(rank, p1, p2, valp1):
    initPerm = [i for i in range(0, len(valp1))]
    if p2 == (p1 + 1)%3:
        for i in range(0, len(valp1)):
            randomPos = np.uint64(generate_random_shareFromAES("next", rank))
            randomPos = np.uint64(randomPos%len(valp1))
            te = initPerm[i]
            initPerm[i] = initPerm[randomPos]
            initPerm[randomPos] = te
            
    else:
        for i in range(0, len(valp1)):
            randomPos =np.uint64(generate_random_shareFromAES("prev", rank))
            randomPos = np.uint64(randomPos%len(valp1))
            te = initPerm[i]
            initPerm[i] = initPerm[randomPos]
            initPerm[randomPos] = te            
    # print("permutation", initPerm)     
    return initPerm

def getLenRandomVals(rank, numBlocks,lenVal, p1, p2):
    randomVals = []
    if p2 == (p1+1)%3:
        for i in range(0, lenVal):
            rv = []
            for j in range(0, numBlocks):
                randomVal = np.uint64(generate_random_shareFromAES("next", rank))
                rv.append(randomVal)
            randomVals.append(rv)
    else:
        for i in range(0, lenVal):
            rv = []
            for j in range(0, numBlocks):
                randomVal = np.uint64(generate_random_shareFromAES("prev", rank))
                rv.append(randomVal)
            randomVals.append(rv)
    currentParty = "party" + str(p1 + 1)
    # setattr(crypten, currentParty, getattr(crypten, currentParty) + 1)
    return randomVals


def applyPermutation(arr, permutation):
    res = []

    for i in range(0, len(permutation)):
        res.append(arr[permutation[i]])

    return res

def shuffleWithRanks(rank, numBlocks, p1, p2, p3, A, B, pi):
    permutation = pi  
    a0 = []
    b0 = []
    c0 = []

    pia = []
    pib = []
    pic = []

    if rank == p1:
        pia = applyPermutation(A, permutation)
        pib = applyPermutation(B, permutation)
        a0 = getLenRandomVals(rank, numBlocks, len(A), p1, p3)
        val = np.array(pia)^np.array(a0)
        dim = val.shape
        modifiedCommunicator.mySend1(val, p2)
        val1 = modifiedCommunicator.myRecieve1(p2, dim)
        b0 = list(np.array(val1)^ val ^ np.array(pib))
        return a0, b0

    elif rank == p2:
        pib = applyPermutation(A, permutation)
        pic = applyPermutation(B, permutation)
        c0 = getLenRandomVals(rank, numBlocks, len(A), p2, p3)
        val = np.array(pic)^np.array(c0)
        dim = val.shape
        val1 = modifiedCommunicator.myRecieve1(p1, dim)
        modifiedCommunicator.mySend1(val, p1)
        b0 = list(np.array(val1)^ val ^ np.array(pib))
        return b0, c0
    else:
        c0 = getLenRandomVals(rank, numBlocks, len(A), p3, p2)
        a0 = getLenRandomVals(rank, numBlocks, len(A), p3, p1)
        return c0, a0

def shuffle(rank, masks):
    #Shuffle needs to be done 3 times
    masks = shuffleWithRanks(rank, 0, 1, 2, masks)
    masks = shuffleWithRanks(rank, 1, 2, 0, masks)
    masks = shuffleWithRanks(rank, 2, 0, 1, masks)
    return masks

def shufflePreProcessing(rank, array):
    if rank == 0:
        for i in range(0, len(array.masks)):
            array.masks[i][0] = array.masks[i][0]^array.share[i]   
    
    elif rank == 2:
        for i in range(0, len(array.masks)):
            array.masks[i][1] = array.masks[i][1]^array.share[i]  
    
    for i in range(0, len(array.share)):
        array.share[i] = np.uint64(0)

    return array

