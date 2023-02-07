#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import sys

sys.path.insert(0, '../../')
sys.path.insert(0, '../SwiftMPC')

from cryptenLocal.SwiftMPC import utils
import cryptenLocal as crypten
import torch
from cryptenLocal.cuda import CUDALongTensor
from Crypto.Cipher import AES
import binascii, os
import pyaes, secrets
import hashlib


iv = secrets.randbits(128)
def generate_random_ring_element(size, ring_size=(2 ** 64), generator=None, **kwargs):
    """Helper function to generate a random number from a signed ring"""
    
    if generator is None:
        device = kwargs.get("device", torch.device("cpu"))
        device = torch.device("cpu") if device is None else device
        device = torch.device(device) if isinstance(device, str) else device
        generator = crypten.generators["local"][device]
    # TODO (brianknott): Check whether this RNG contains the full range we want.
    rand_element = torch.randint(
        -(ring_size // 2),
        (ring_size - 1) // 2,
        size,
        generator=generator,
        dtype=torch.long,
        **kwargs
    )
    if rand_element.is_cuda:
        return CUDALongTensor(rand_element)
    return rand_element


def generate_random_shareFromAES(generatorPos, rank, ring_size=(2 ** 64),**kwargs):
    """Helper function to generate a random number from a signed ring"""
    
    generator = crypten.generators[generatorPos][2:32]

    generator = generator + "00"
    # print(rank, generator)
    
    counter = 0
    posWith = 0

    if generatorPos == "prev":
        posWith = (rank - 1)%3
    elif generatorPos == "next":
        posWith = (rank + 1)%3
    else:
        posWith = 3


    #Getting the counter for the current party
    if posWith == 0:
        counter = utils.party1
        utils.party1 = utils.party1 + 1
    elif posWith == 1:
        counter = utils.party2
        utils.party2 = utils.party2 + 1
    elif posWith == 2:
        counter = utils.party3
        utils.party3 = utils.party3 + 1
    else:
        counter = utils.globalCounter
        utils.globalCounter = utils.globalCounter + 1
    

    if generatorPos == "premult":
        counter = crypten.premult
    # print(rank, counter, bytes.fromhex(generator))
    aes = pyaes.AESModeOfOperationCTR(bytes.fromhex(generator), pyaes.Counter(iv))
    myString = str(counter)
    encryptedVal = binascii.hexlify(aes.encrypt(hashlib.sha256(myString.encode('utf-8')).hexdigest()))
    # print(encryptedVal, rank)
    ciphertext = int(encryptedVal, 16)%ring_size - (2**63)
    
    return ciphertext

def generate_kbit_random_tensor(size, bitlength=None, generator=None, **kwargs):
    """Helper function to generate a random k-bit number"""
    if bitlength is None:
        bitlength = torch.iinfo(torch.long).bits
    if bitlength == 64:
        return generate_random_ring_element(size, generator=generator, **kwargs)
    if generator is None:
        device = kwargs.get("device", torch.device("cpu"))
        device = torch.device("cpu") if device is None else device
        device = torch.device(device) if isinstance(device, str) else device
        generator = crypten.generators["local"][device]
    rand_tensor = torch.randint(
        0, 2 ** bitlength, size, generator=generator, dtype=torch.long, **kwargs
    )
    if rand_tensor.is_cuda:
        return CUDALongTensor(rand_tensor)
    return rand_tensor

