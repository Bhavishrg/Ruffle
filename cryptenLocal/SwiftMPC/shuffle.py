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
    print("permutation", initPerm)     
    return initPerm

def getLenRandomVals(rank, lenVal, p1, p2):
    randomVals = []
    if p2 == (p1+1)%3:
        for i in range(0, lenVal):
            randomVal = np.uint64(generate_random_shareFromAES("next", rank))
            randomVals.append(randomVal)
    else:
        for i in range(0, lenVal):
            randomVal = np.uint64(generate_random_shareFromAES("prev", rank))
            randomVals.append(randomVal)
    currentParty = "party" + str(p1 + 1)
    setattr(crypten, currentParty, getattr(crypten, currentParty) + 1)
    return randomVals

def applyPermutation(arr, permutation):
    res = []

    for i in range(0, len(permutation)):
        res.append(arr[permutation[i]])

    return res

def shuffleWithRanks(rank, p1, p2, p3, masks):
    if rank == p1:
        permutation = getRandomPermutation(rank, p1, p2, masks)   
        masks = applyPermutation(masks, permutation)
    elif rank == p2:
        permutation = getRandomPermutation(rank, p2, p1, masks)   
        masks = applyPermutation(masks, permutation)
 

    a0 = []
    b0 = []
    c0 = []

    if rank == p1:
        a0 = getLenRandomVals(rank, len(masks), p1, p3)
    elif rank == p2:
        c0 = getLenRandomVals(rank, len(masks), p2, p3)
    else:
        c0 = getLenRandomVals(rank, len(masks), p3, p2)
        a0 = getLenRandomVals(rank, len(masks), p3, p1)

 
    pia = []
    pib = []
    pic = []

    #Send from p1 to p2  
    pia = [masks[i][0] for i in range(0,len(masks))]

    if rank == p1:
        if p2 == (p1+1)%3:
            pib = [masks[i][1] for i in range(0,len(masks))]
        else:
            pib = [masks[i][0] for i in range(0,len(masks))]
    else:
        if p1 == (p2 + 1)%3:
            pib = [masks[i][1] for i in range(0,len(masks))]
        else:
            pib = [masks[i][0] for i in range(0,len(masks))]
    
    if rank == p1 or rank == p3:
        for i in range(0, len(pia)):
            pia[i] = pia[i]^a0[i]
        
    for i in range(0, len(pia)):
        if rank == p1:
            modifiedCommunicator.mySend(pia[i], p2)
        elif rank == p2:
            pia[i] = modifiedCommunicator.myRecieve(p1)  

    #Send from p2 to p1
    pic = [masks[i][1] for i in range(0,len(masks))]

    if rank == p2 or rank == p3:
        for i in range(0, len(pic)):
            pic[i] = pic[i]^c0[i]
    
    for i in range(0, len(pic)):
        if rank == p2:
            modifiedCommunicator.mySend(pic[i], p1)
        elif rank == p1:
            pic[i] = modifiedCommunicator.myRecieve(p2)

    if rank == p1 or rank == p2:
        for i in range(0, len(pia)):
            b0.append(pib[i]^pia[i]^pic[i])

    returnVal = []

 
    for i in range(0, max(len(a0), len(b0))):
        if rank == p1:
            if p2 == (p1 + 1)%3:
                returnVal.append([a0[i], b0[i]])
            else:
                returnVal.append([b0[i], a0[i]])
        elif rank == p2:
            if p3 == (p2 + 1)%3:
                returnVal.append([b0[i], c0[i]])
            else:
                returnVal.append([c0[i], b0[i]])
        else:
            if p1 == (p3+1)%3:
                returnVal.append([c0[i], a0[i]])
            else:
                returnVal.append([a0[i], c0[i]])
    return returnVal

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