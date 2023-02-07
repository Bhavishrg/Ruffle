import pyaes
import cryptenLocal as crypten
import binascii
import hashlib
#Counter wrt to other parties
party1 = 0
party2 = 0
party3 = 0
globalCounter = 0
# iv = "hellow"

# def generate_random_shareFromAES(generatorPos, rank, ring_size=(2 ** 64),**kwargs):
#     """Helper function to generate a random number from a signed ring"""
    
#     generator = crypten.generators[generatorPos][2:32]

#     generator = generator + "00"
#     # print(rank, generator)
    
#     counter = 0
#     posWith = 0

#     if generatorPos == "prev":
#         posWith = (rank - 1)%3
#     else:
#         posWith = (rank + 1)%3

#     #Getting the counter for the current party
#     if posWith == 0:
#         counter = party1
#         party1 =  party1 + 1
#     elif posWith == 1:
#         counter = party2
#         party2 = party2 + 1
#     else:
#         counter = party3
#         party3 = party3 + 1

#     if generatorPos == "premult":
#         counter = crypten.premult
#     # print(rank, counter, bytes.fromhex(generator))
#     aes = pyaes.AESModeOfOperationCTR(bytes.fromhex(generator), pyaes.Counter(iv))
#     myString = str(counter)
#     encryptedVal = binascii.hexlify(aes.encrypt(hashlib.sha256(myString.encode('utf-8')).hexdigest()))
#     # print(encryptedVal, rank)
#     ciphertext = int(encryptedVal, 16)%ring_size - (2**63)
    
#     return ciphertext