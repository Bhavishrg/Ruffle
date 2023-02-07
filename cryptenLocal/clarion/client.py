from ast import mod
from glob import glob
import sys


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

encryptionKey = ""
key1 = ""
key2 = ""

def int_of_string(s):
    return int(binascii.hexlify(s), 16)

def encrypt_message(key, plaintext):
    iv = (0).to_bytes(16, byteorder = 'big')
    ctr = Counter.new(128, initial_value=int_of_string(iv))
    aes = AES.new(key, AES.MODE_CTR, counter=ctr)
    return aes.encrypt(plaintext)

def decrypt_message(key, ciphertext):
    iv = (0).to_bytes(16, byteorder = 'big')
    ctr = Counter.new(128, initial_value=int_of_string(iv))
    aes = AES.new(key, AES.MODE_CTR, counter=ctr)
    return aes.decrypt(ciphertext[16:])

def createMessage(rank, numBlocks):
    #Number of bytes per block is 16
    numBytesPerBlock = 16

    totalBytes = numBytesPerBlock*numBlocks

    messageValue = b""

    for i in range(0, totalBytes):
        messageValue = messageValue + (97 + rank).to_bytes(1, byteorder = 'big')
    
    return messageValue

def encryptUsingAES(messageValue):
    global encryptionKey
    encryptionKey = os.urandom(16)
    encryptedMessage = encrypt_message(encryptionKey, messageValue)
    return encryptedMessage

def sampleSeeds():
    global key1, key2
    key1 = os.urandom(16)
    key2 = os.urandom(16)

#expand a seed using aes in CTR mode
def AesPRG(msgLen, seed):
    
    msgLen = msgLen*16
    ct = b""

    for i in range(0, msgLen):
        ct = ct + (0).to_bytes(1, byteorder = 'big')
    
    expandedKey = encrypt_message(seed, ct)
    
    return expandedKey



def clientSimulation(rank, numBlocks):
    
    #Create the message
    messageValue = createMessage(rank, numBlocks)

    #Encrypt the message
    encryptedMessage = encryptUsingAES(messageValue)

    #Sample random values for keys 
    sampleSeeds()
    expandedkey1 = AesPRG(numBlocks + 1, key1)
    expandedkey2 = AesPRG(numBlocks + 1, key2)

    keyShares = []

    for i in range(0, numBlocks + 1):
        val1 = int.from_bytes(expandedkey1[16*i:16*i + 16], byteorder = 'big')
        val2 = int.from_bytes(expandedkey2[16*i:16*i + 16], byteorder = 'big')
        keyShares.append(modp.addp(val1, val2))
      
    tag = 0
 
    for i in range(0, numBlocks):
        val = int.from_bytes(encryptedMessage[16*i:16*i + 16], byteorder = 'big')
        tag = modp.addp(tag, modp.multp(val, keyShares[i]))

    encryptionKeyAsInt = int.from_bytes(encryptionKey, byteorder = 'big')
    tag = modp.addp(tag, modp.multp(keyShares[numBlocks], encryptionKeyAsInt))

   
    #Get shares
    [tagShare1, tagShare2] = modp.getAdditiveSecretShares(tag)
    [ek1, ek2] = modp.getAdditiveSecretShares(encryptionKeyAsInt)
    secretSharedct = []
    for i in range(0, numBlocks):
        val = int.from_bytes(encryptedMessage[16*i:16*i + 16], byteorder = 'big')
        [share1, share2] = modp.getAdditiveSecretShares(val)
        secretSharedct.append([share1, share2])

    #Send to servers 1 and 2
    modifiedCommunicator.mySend(key1, 0)
    modifiedCommunicator.mySend(tagShare1, 0)
    for i in range(0, numBlocks):
        modifiedCommunicator.mySend(secretSharedct[i][0], 0)
    modifiedCommunicator.mySend(ek1, 0)

    modifiedCommunicator.mySend(key2, 1)
    modifiedCommunicator.mySend(tagShare2, 1)
    for i in range(0, numBlocks):
        modifiedCommunicator.mySend(secretSharedct[i][1], 1)
    modifiedCommunicator.mySend(ek2, 1)

    