from ast import mod
import sys
from unicodedata import ucd_3_2_0
import time


sys.path.insert(0, '../../')

from ab.shuffle import *
from cryptenLocal.SwiftMPC.ouputCommitments import getFullCommitmentHash, convertHexHashToInteger
from ecdsa import SigningKey



def generateRandomness(rank, numBlocks, numClients):
    prev = (rank-1)%3
    next = (rank+1)%3
    sharePrev = getLenRandomVals(rank, numBlocks, numClients, rank, prev)
    shareNext = getLenRandomVals(rank, numBlocks, numClients, rank, next)
    return sharePrev, shareNext

def commitments(rank, val1, val2):
    prev = (rank-1)%3
    next = (rank+1)%3
    comPrev = getFullCommitmentHash(val1)
    comNext = getFullCommitmentHash(val2)

    if rank == 0:
        c0 = comPrev
        c1 = comNext
        modifiedCommunicator.mySend1(comNext, next)
        c2 = modifiedCommunicator.myRecieve1(prev, [1,4])[0]

        modifiedCommunicator.mySend1(comPrev, prev)
        c2_ = modifiedCommunicator.myRecieve1(next, [1,4])[0]
        if (c2 == c2_).any():
            print("match")
    elif rank == 1:
        c1 = comPrev
        c2 = comNext
        c0 = modifiedCommunicator.myRecieve1(prev, [1,4])[0]
        modifiedCommunicator.mySend1(comNext, next)

        c0_ = modifiedCommunicator.myRecieve1(next, [1,4])[0]
        modifiedCommunicator.mySend1(comPrev, prev)
        if (c0 == c0_).any():
            print("match")
    else:
        c2 = comPrev
        c0 = comNext
        c1 = modifiedCommunicator.myRecieve1(prev, [1,4])[0]
        modifiedCommunicator.mySend1(comNext, next)

        c1_ = modifiedCommunicator.myRecieve1(next, [1,4])[0]
        modifiedCommunicator.mySend1(comPrev, prev)

        if (c1 == c1_).any():
            print("match")
    c = getFullCommitmentHash(list(c0)+list(c1)+list(c2))
    return c

def sendtoclients(rank, val1, val2, c):
    modifiedCommunicator.mySend1(c, 3)
    modifiedCommunicator.mySend1(val1 + val2, 3)

def generateMsg(rank, numBlocks, numClients):
    
    return 1

def clientSim(rank, numBlocks, numClients):
    c0 =  modifiedCommunicator.myRecieve1(0,[1,4])
    c1 =  modifiedCommunicator.myRecieve1(1,[1,4])
    c2 =  modifiedCommunicator.myRecieve1(2,[1,4])
    # print(c0, c1, c2)
    al0 = modifiedCommunicator.myRecieve1(0, [2*numClients, numBlocks])[:numClients]
    al1 = modifiedCommunicator.myRecieve1(1, [2*numClients, numBlocks])[:numClients]
    al2 = modifiedCommunicator.myRecieve1(2, [2*numClients, numBlocks])[:numClients]
    
    timeStart = time.time()

    al0 = np.array(al0)
    al1 = np.array(al1)
    al2 = np.array(al2)

    beta = al0^al1^al2
    timeEnd = time.time()
    totime =  timeEnd - timeStart

    modifiedCommunicator.mySend1(beta, 0)
    modifiedCommunicator.mySend1(beta, 1)
    modifiedCommunicator.mySend1(beta, 2)
    print("Client simulation is done")
    return totime


def shuffleBen(rank, numBlocks, a, b, pia, pib, ka, kb):
    numBlocks = numBlocks + 1
    if rank == 0:  
        pi0 = pia
        pi1 = pib
        al0 = a
        al1 = b
        k0 = ka
        k1 = kb
        

        al0 = list(np.concatenate((np.array(al0), np.array(k0)), axis=1))
        al1 = list(np.concatenate((np.array(al1), np.array(k1)), axis=1))

        #r1
        al0_, al1_ = shuffleWithRanks(rank, numBlocks, 2, 0, 1, al0, al1, pi0)
        set_equality(rank, numBlocks, len(a), al0, al1, al0_, al1_)
        #r2
        al0, al1 = shuffleWithRanks(rank, numBlocks, 0, 1, 2, al0_, al1_, pi1)
        set_equality(rank, numBlocks, len(a), al0_, al1_, al0, al1)
        #r3
        a, b = shuffleWithRanks(rank, numBlocks, 1, 2, 0, al0, al1, pi1)
        set_equality(rank, numBlocks, len(a), al0, al1, a, b)

    elif rank == 1:  
        pi1 = pia
        pi2 = pib
        al1 = a
        al2 = b
        k1 = ka
        k2 = kb

        al1 = list(np.concatenate((np.array(al1), np.array(k1)), axis=1))
        al2 = list(np.concatenate((np.array(al2), np.array(k2)), axis=1))

        #r1
        al1_, al2_ = shuffleWithRanks(rank, numBlocks, 2, 0, 1, al1, al2, pi1)
        set_equality(rank, numBlocks, len(a), al1, al2, al1_, al2_)
        #r2
        al1, al2 = shuffleWithRanks(rank, numBlocks, 0, 1, 2, al1_, al2_, pi1)
        set_equality(rank, numBlocks, len(a), al1_, al2_, al1, al2)
        #r3
        a, b = shuffleWithRanks(rank, numBlocks, 1, 2, 0, al1, al2, pi2)
        set_equality(rank, numBlocks, len(a), al1, al2, a, b)
    
    elif rank == 2:  
        pi2 = pia
        pi0 = pib
        al2 = a
        al0 = b
        k2 = ka
        k0 = kb

        al2 = list(np.concatenate((np.array(al2), np.array(k2)), axis=1))
        al0 = list(np.concatenate((np.array(al0), np.array(k0)), axis=1))

        #r1
        al2_, al0_ = shuffleWithRanks(rank, numBlocks, 2, 0, 1, al2, al0, pi0)
        set_equality(rank, numBlocks, len(a), al2, al0, al2_, al0_)
        #r2
        al2, al0 = shuffleWithRanks(rank, numBlocks, 0, 1, 2, al2_, al0_, pi0)
        set_equality(rank, numBlocks, len(a), al2_, al0_, al2, al0)
        #r3
        a, b = shuffleWithRanks(rank, numBlocks, 1, 2, 0, al2, al0, pi2)
        set_equality(rank, numBlocks, len(a), al2, al0, a, b)
    a, b = np.array(a), np.array(b)
    return list(a[:,:-1]), list(b[:,:-1])


def shufflePre(rank, numBlocks, a, b, ra, rb, pia, pib, ka, kb):
    numBlocks = numBlocks+1
    if rank == 0:  
        pi0 = pia
        pi1 = pib
        al0 = a
        al1 = b
        r0 = ra
        r1 = rb
        k0 = ka
        k1 = kb
            
        r0 = list(np.concatenate((np.array(r0), np.array(k0)), axis=1))
        r1 = list(np.concatenate((np.array(r1), np.array(k1)), axis=1))
            
        al0 = list(np.concatenate((np.array(al0), np.array(k0)), axis=1))
        al1 = list(np.concatenate((np.array(al1), np.array(k1)), axis=1))

        
        #r1
        al0 = list(np.array(al0) ^ np.array(r0))
        al0_, al1_ = shuffleWithRanks(rank, numBlocks, 2, 0, 1, al0, al1, pi0)
        set_equality(rank, numBlocks, len(a), al0, al1, al0_, al1_)
        #r2
        al1_ = list(np.array(al1_) ^ np.array(r1))
        al0, al1 = shuffleWithRanks(rank, numBlocks, 0, 1, 2, al0_, al1_, pi1)
        set_equality(rank, numBlocks, len(a), al0_, al1_, al0, al1)
        #r3
        a, b = shuffleWithRanks(rank, numBlocks, 1, 2, 0, al0, al1, pi1)
        set_equality(rank, numBlocks, len(a), al0, al1, a, b)

    elif rank == 1:  
        pi1 = pia
        pi2 = pib
        al1 = a
        al2 = b
        r1 = ra
        r2 = rb
        k1 = ka
        k2 = kb

        r1 = list(np.concatenate((np.array(r1), np.array(k1)), axis=1))
        r2 = list(np.concatenate((np.array(r2), np.array(k2)), axis=1))
        al1 = list(np.concatenate((np.array(al1), np.array(k1)), axis=1))
        al2 = list(np.concatenate((np.array(al2), np.array(k2)), axis=1))

        #r1
        al1_, al2_ = shuffleWithRanks(rank, numBlocks, 2, 0, 1, al1, al2, pi1)
        set_equality(rank, numBlocks, len(a), al1, al2, al1_, al2_)
        #r2
        al1_ = list(np.array(al1_) ^ np.array(r1))
        al1, al2 = shuffleWithRanks(rank, numBlocks, 0, 1, 2, al1_, al2_, pi1)
        set_equality(rank, numBlocks, len(a), al1_, al2_, al1, al2)
        #r3
        al2 = list(np.array(al2) ^ np.array(r2))
        a, b = shuffleWithRanks(rank, numBlocks, 1, 2, 0, al1, al2, pi2)
        set_equality(rank, numBlocks, len(a), al1, al2, a, b)
    
    elif rank == 2:  
        pi2 = pia
        pi0 = pib
        al2 = a
        al0 = b
        r2 = ra
        r0 = rb
        k2 = ka
        k0 = kb

        r2 = list(np.concatenate((np.array(r2), np.array(k2)), axis=1))
        r0 = list(np.concatenate((np.array(r0), np.array(k0)), axis=1))

        al2 = list(np.concatenate((np.array(al2), np.array(k2)), axis=1))
        al0 = list(np.concatenate((np.array(al0), np.array(k0)), axis=1))

        #r1
        al0 = list(np.array(al0) ^ np.array(r0))
        al2_, al0_ = shuffleWithRanks(rank, numBlocks, 2, 0, 1, al2, al0, pi0)
        set_equality(rank, numBlocks, len(a), al2, al0, al2_, al0_)
        #r2
        al2, al0 = shuffleWithRanks(rank, numBlocks, 0, 1, 2, al2_, al0_, pi0)
        set_equality(rank, numBlocks, len(a), al2_, al0_, al2, al0)
        #r3
        al2 = list(np.array(al2) ^ np.array(r2))
        a, b = shuffleWithRanks(rank, numBlocks, 1, 2, 0, al2, al0, pi2)
        set_equality(rank, numBlocks, len(a), al2, al0, a, b)
    a, b = np.array(a), np.array(b)
    return list(a[:,:-1]), list(b[:,:-1])



def broadcast_ints(rank, msg, sk, vk0, vk1, vk2):
    ## add verification
    if rank == 0:
        #round 1
        signature = sk.sign(bytes(str(msg), 'utf-8'))
        signature = signature.hex()
        int_sign0 = [int(signature[:16],16), int(signature[16:32],16), int(signature[32:48],16), int(signature[48:64],16), int(signature[64:80],16), int(signature[80:96],16)]
        int_sign1 = recoverKey(modifiedCommunicator.myRecieve1(1, [1,6])[0])
        modifiedCommunicator.mySend1(int_sign0, 1)
        modifiedCommunicator.mySend1(int_sign0, 2)
        int_sign2 = recoverKey(modifiedCommunicator.myRecieve1(2, [1,6])[0])
        msg1 = modifiedCommunicator.myRecieve1(1, [len(msg),len(msg[0])])[0]
        modifiedCommunicator.mySend1(msg, 1)
        modifiedCommunicator.mySend1(msg, 2)
        msg2 = modifiedCommunicator.myRecieve1(2, [len(msg),len(msg[0])])[0]

        #round 2
        signature1 = sk.sign(bytes(str(int_sign1), 'utf-8')).hex()
        signature2 = sk.sign(bytes(str(int_sign2), 'utf-8')).hex()
        int_sign1 = [int(signature1[:16],16), int(signature1[16:32],16), int(signature1[32:48],16), int(signature1[48:64],16), int(signature1[64:80],16), int(signature1[80:96],16)]
        int_sign2 = [int(signature2[:16],16), int(signature2[16:32],16), int(signature2[32:48],16), int(signature2[48:64],16), int(signature2[64:80],16), int(signature2[80:96],16)]
        int_sign2_ = recoverKey(modifiedCommunicator.myRecieve1(1, [1,6])[0])
        modifiedCommunicator.mySend1(int_sign2, 1)
        modifiedCommunicator.mySend1(int_sign1, 2)
        int_sign1 = recoverKey(modifiedCommunicator.myRecieve1(2, [1,6])[0])
        
        # msg2 = modifiedCommunicator.myRecieve1(1, [len(msg),len(msg[0])])[0]
        
        # modifiedCommunicator.mySend1(msg2, 1)
        
        # modifiedCommunicator.mySend1(msg1, 2)
        # msg1 = modifiedCommunicator.myRecieve1(2, [len(msg),len(msg[0])])[0]
        
    
    if rank == 1:
        #round 1
        signature = sk.sign(bytes(str(msg), 'utf-8'))
        signature = signature.hex()
        int_sign1 = [int(signature[:16],16), int(signature[16:32],16), int(signature[32:48],16), int(signature[48:64],16), int(signature[64:80],16), int(signature[80:96],16)]
        modifiedCommunicator.mySend1(int_sign1, 2)
        modifiedCommunicator.mySend1(int_sign1, 0)
        int_sign0 = recoverKey(modifiedCommunicator.myRecieve1(0, [1,6])[0])
        int_sign2 = recoverKey(modifiedCommunicator.myRecieve1(2, [1,6])[0])    
        modifiedCommunicator.mySend1(msg, 2)
        modifiedCommunicator.mySend1(msg, 0)
        msg0 = modifiedCommunicator.myRecieve1(0, [len(msg),len(msg[0])])[0]
        msg2 = modifiedCommunicator.myRecieve1(2, [len(msg),len(msg[0])])[0]

        #round 2
        signature2 = sk.sign(bytes(str(int_sign2), 'utf-8')).hex()
        signature0 = sk.sign(bytes(str(int_sign0), 'utf-8')).hex()
        int_sign0 = [int(signature0[:16],16), int(signature0[16:32],16), int(signature0[32:48],16), int(signature0[48:64],16), int(signature0[64:80],16), int(signature0[80:96],16)]
        int_sign2 = [int(signature2[:16],16), int(signature2[16:32],16), int(signature2[32:48],16), int(signature2[48:64],16), int(signature2[64:80],16), int(signature2[80:96],16)]
        modifiedCommunicator.mySend1(int_sign0, 2)
        modifiedCommunicator.mySend1(int_sign2, 0)
        int_sign0_ = recoverKey(modifiedCommunicator.myRecieve1(0, [1,6])[0])
        int_sign2_ = recoverKey(modifiedCommunicator.myRecieve1(2, [1,6])[0])
        # modifiedCommunicator.mySend1(msg0, 0)
        # modifiedCommunicator.mySend1(msg2, 2)
        # print("sfsfvs")
        # msg2 = modifiedCommunicator.myRecieve1(0, [len(msg),len(msg[0])])[0]
        
        # msg0 = modifiedCommunicator.myRecieve1(2, [len(msg),len(msg[0])])[0]


    if rank == 2:
        # round 1
        signature = sk.sign(bytes(str(msg), 'utf-8'))
        signature = signature.hex()
        int_sign2 = [int(signature[:16],16), int(signature[16:32],16), int(signature[32:48],16), int(signature[48:64],16), int(signature[64:80],16), int(signature[80:96],16)]
        int_sign1 = recoverKey(modifiedCommunicator.myRecieve1(1, [1,6])[0])
        int_sign0 = recoverKey(modifiedCommunicator.myRecieve1(0, [1,6])[0])
        modifiedCommunicator.mySend1(int_sign2, 0)
        modifiedCommunicator.mySend1(int_sign2, 1)

        msg1 = modifiedCommunicator.myRecieve1(1, [len(msg),len(msg[0])])[0]
        msg0 = modifiedCommunicator.myRecieve1(0, [len(msg),len(msg[0])])[0]
        modifiedCommunicator.mySend1(msg, 0)
        modifiedCommunicator.mySend1(msg, 1)

        #round 2
        signature1 = sk.sign(bytes(str(int_sign1), 'utf-8')).hex()
        signature0 = sk.sign(bytes(str(int_sign2), 'utf-8')).hex()
        int_sign1 = [int(signature1[:16],16), int(signature1[16:32],16), int(signature1[32:48],16), int(signature1[48:64],16), int(signature1[64:80],16), int(signature1[80:96],16)]
        int_sign0 = [int(signature0[:16],16), int(signature0[16:32],16), int(signature0[32:48],16), int(signature0[48:64],16), int(signature0[64:80],16), int(signature0[80:96],16)]
        
        int_sign1_ = recoverKey(modifiedCommunicator.myRecieve1(1, [1,6])[0])
        int_sign0_ = recoverKey(modifiedCommunicator.myRecieve1(0, [1,6])[0])
        modifiedCommunicator.mySend1(int_sign1, 0)
        modifiedCommunicator.mySend1(int_sign0, 1)
        
        # msg0 = modifiedCommunicator.myRecieve1(1, [len(msg),len(msg[0])])[0]
        
        # msg1 = modifiedCommunicator.myRecieve1(0, [len(msg),len(msg[0])])[0]
        # print("sfsfvs")
        # modifiedCommunicator.mySend1(msg1, 0)
        # modifiedCommunicator.mySend1(msg0, 1)
        
        
    return 1



def generateBeta(rank, numBlocks, numClients):
    beta = modifiedCommunicator.myRecieve1(3, [numClients, numBlocks])
    return beta

def ShuffleOnl(rank, numBlocks, beta, ra, rb, pia, pib):
    if rank == 0:
        pi0 = pia
        pi1 = pib
        r0 = ra
        r1 = rb
        #round 1
        beta0 = list(np.array(beta).astype("uint64")^np.array(r0))
        beta0 = applyPermutation(beta0, pi0)
        hbeta0 = getFullCommitmentHash(beta0)
        modifiedCommunicator.mySend1(hbeta0, 1)
        beta1 = list(np.array(beta0).astype("uint64")^np.array(r1))
        beta1 = applyPermutation(beta1, pi1)
        hbeta1 = getFullCommitmentHash(beta1)
        modifiedCommunicator.mySend1(hbeta1, 2)
        # round 2
        beta_ = modifiedCommunicator.myRecieve1(2, [len(beta), numBlocks])
        hbeta_ = modifiedCommunicator.myRecieve1(1, [1,4])
        hbeta = getFullCommitmentHash(beta_)
        if (hbeta_ == hbeta).all():
            print('match')
        
    
    if rank == 1:
        pi1 = pia
        pi2 = pib
        r1 = ra
        r2 = rb
        #round 1
        hbeta0 = modifiedCommunicator.myRecieve1(0, [1,4])
        beta0 = modifiedCommunicator.myRecieve1(2, [len(beta), numBlocks])
        hbeta0_ = getFullCommitmentHash(beta0)

        if (hbeta0 == hbeta0_).all():
            print("match")
        #round 2 
        beta1 = list(np.array(beta0).astype("uint64")^np.array(r1))
        modifiedCommunicator.mySend1(beta1, 2)
        beta2 = list(np.array(beta1).astype("uint64")^np.array(r2))
        beta_ = applyPermutation(beta2, pi2)
        hbeta_ = getFullCommitmentHash(beta_)
        modifiedCommunicator.mySend1(hbeta_, 0)
    
        
    if rank == 2:
        pi2 = pia
        pi0 = pib
        r2 = ra
        r0 = rb
        
        beta0 = list(np.array(beta).astype("uint64")^np.array(r0).astype('uint64'))
        beta0 = applyPermutation(beta0, pi0)
        modifiedCommunicator.mySend1(beta0, 1)

        hbeta1 = modifiedCommunicator.myRecieve1(0, [1,4])
        beta1 = modifiedCommunicator.myRecieve1(1, [len(beta), numBlocks])
        hbeta1_ = getFullCommitmentHash(beta1)
        if (hbeta1 == hbeta1_).all():
            print('match')
        beta2 = list(np.array(beta1).astype("uint64")^np.array(r2))
        beta_ = applyPermutation(beta2, pi2)
        modifiedCommunicator.mySend1(beta_, 0)
    return beta_

def reconstOutput(rank,beta, ala, alb):
    if rank==0:
        modifiedCommunicator.mySend1(ala,1)
        halb = getFullCommitmentHash(alb)
        modifiedCommunicator.mySend1(halb, 2)
        alc = modifiedCommunicator.myRecieve1(2, [len(ala),len(ala[0])])
        halc = modifiedCommunicator.myRecieve1(1, [1,4])[0]
        halc_ = getFullCommitmentHash(alc)
        if list(halc) == halc_:
            print("match")


    if rank==1:
        alc = modifiedCommunicator.myRecieve1(0, [len(ala),len(ala[0])])
        modifiedCommunicator.mySend1(ala,2)
        halb = getFullCommitmentHash(alb)
        modifiedCommunicator.mySend1(halb, 0)
        halc = modifiedCommunicator.myRecieve1(2, [1,4])[0]
        halc_ = getFullCommitmentHash(alc)
        if list(halc) == halc_:
            print("match")

        

    if rank==2:
        alc = modifiedCommunicator.myRecieve1(1, [len(ala),len(ala[0])])
        halc = modifiedCommunicator.myRecieve1(0, [1,4])[0]
        modifiedCommunicator.mySend1(ala,0)
        halb = getFullCommitmentHash(alb)
        modifiedCommunicator.mySend1(halb, 1)
        halc_ = getFullCommitmentHash(alc)
        if list(halc) == halc_:
            print("match")
    return np.array(beta) ^ np.array(ala) 


def recoverKey(key):
    vk = ''
    for i in key:
        vk += bin(i)[2:]
    return int(vk, 2).to_bytes((len(vk) + 7) // 8, 'big')


def dotprod(rank, numClients, ua, ub, va, vb):
    ra, rb = generateRandomness(rank, numClients, 1)
    if rank == 0:
        z0 = (np.array(ra) - np.array(rb))[0]%2
        u0 = np.array(ua).astype("bool")
        u1 = np.array(ub).astype("bool")
        v0 = np.array(va).astype("bool")
        v1 = np.array(vb).astype("bool")
        y0 = int(sum((u0 & v1) ^ (u1 & v0) ^ (u1 & v1) ^ z0)%2)
        modifiedCommunicator.mySend(y0, 2)
        y1 = modifiedCommunicator.myRecieve(1)
        return y0, y1


    if rank == 1:
        z1 = (np.array(ra) - np.array(rb))[0]%2
        u1 = np.array(ua).astype("bool")
        u2 = np.array(ub).astype("bool")
        v1 = np.array(va).astype("bool")
        v2 = np.array(vb).astype("bool")
        y1 = int(sum((u1 & v2) ^ (u2 & v1) ^ (u2 & v2) ^ z1)%2)
        y2 = modifiedCommunicator.myRecieve(2)
        modifiedCommunicator.mySend1(y1, 0)
        return y1, y2
        
    if rank == 2:
        z2 = (np.array(ra) - np.array(rb))[0]%2
        u2 = np.array(ua).astype("bool")
        u0 = np.array(ub).astype("bool")
        v2 = np.array(va).astype("bool")
        v0 = np.array(vb).astype("bool")
        y2 = int(sum((u2 & v0) ^ (u0 & v2) ^ (u0 & v0) ^ z2)%2)
        y0 = modifiedCommunicator.myRecieve(0)
        modifiedCommunicator.mySend(y2,1)
        return y2, y0
        

def set_equality(rank, numBlocks, numClients, ta, tb, ta_, tb_):
    if rank == 0:
        c = np.ones((numClients, numBlocks)).astype('uint64')
        t0 = np.array(ta)
        t1 = np.array(tb)
        t0_ = np.array(ta_)
        t1_ = np.array(tb_)
        tl0 = np.sum(t0 & c, axis = 1)%2
        tl1 = np.sum(t1 & c, axis = 1)%2
        k0 = np.array([i[-1]%2 for i in t0])
        k1 = np.array([i[-1]%2 for i in t1])

        tl0_ = np.sum(t0 & c, axis = 1)%2
        tl1_ = np.sum(t1 & c, axis = 1)%2
        k0_ = np.array([i[-1]%2 for i in t0_])
        k1_ = np.array([i[-1]%2 for i in t1_])
        for i in range(2):
            sentStart = modifiedCommunicator.totalBitsSent
            sentStart0 = modifiedCommunicator.numberBitsSent0
            sentStart1 = modifiedCommunicator.numberBitsSent1
            sentStart2 = modifiedCommunicator.numberBitsSent2
            recievedStart = modifiedCommunicator.totalBitsRecieved
            recievedStart0 = modifiedCommunicator.numberBitsRecieved0
            recievedStart1 = modifiedCommunicator.numberBitsRecieved1
            recievedStart2 = modifiedCommunicator.numberBitsRecieved2

            v0, v1 = dotprod(rank, numClients, tl0, tl1, k0, k1)
            v0_, v1_ = dotprod(rank, numClients, tl0_, tl1_, k0_, k1_)
            v0 = v0 ^ v0_
            v1 = v1 ^ v1_ 
            reconstOutput(rank, [[0]], [[v0]], [[v1]])

            sentEnd = modifiedCommunicator.totalBitsSent
            sentEnd0 = modifiedCommunicator.numberBitsSent0
            sentEnd1 = modifiedCommunicator.numberBitsSent1
            sentEnd2 = modifiedCommunicator.numberBitsSent2
            recievedEnd = modifiedCommunicator.totalBitsRecieved
            recievedEnd0 = modifiedCommunicator.numberBitsRecieved0
            recievedEnd1 = modifiedCommunicator.numberBitsRecieved1
            recievedEnd2 = modifiedCommunicator.numberBitsRecieved2
            modifiedCommunicator.totalBitsSent += (sentEnd - sentStart)*(46/64)
            modifiedCommunicator.totalBitsRecieved += (recievedEnd - recievedStart)*(46/64)
            modifiedCommunicator.numberBitsSent0 += (sentEnd0 - sentStart0)*46
            modifiedCommunicator.numberBitsRecieved0 += (recievedEnd0 - recievedStart0)*(46/64)
            modifiedCommunicator.numberBitsSent1 += (sentEnd1 - sentStart1)*46
            modifiedCommunicator.numberBitsRecieved1 += (recievedEnd1 - recievedStart1)*(46/64)
            modifiedCommunicator.numberBitsSent2 += (sentEnd2 - sentStart2)*46
            modifiedCommunicator.numberBitsRecieved2 += (recievedEnd2 - recievedStart2)*(46/64)
            



    if rank == 1:
        c = np.ones((numClients, numBlocks)).astype('uint64')
        t1 = np.array(ta)
        t2 = np.array(tb)
        t1_ = np.array(ta_)
        t2_ = np.array(tb_)
        tl1 = np.sum(t1 & c, axis = 1)%2
        tl2 = np.sum(t2 & c, axis = 1)%2
        k1 = np.array([i[-1]%2 for i in t1])
        k2 = np.array([i[-1]%2 for i in t2])

        tl1_ = np.sum(t1 & c, axis = 1)%2
        tl2_ = np.sum(t2 & c, axis = 1)%2
        k1_ = np.array([i[-1]%2 for i in t1_])
        k2_ = np.array([i[-1]%2 for i in t2_])
        for i in range(2):
            sentStart = modifiedCommunicator.totalBitsSent
            sentStart0 = modifiedCommunicator.numberBitsSent0
            sentStart1 = modifiedCommunicator.numberBitsSent1
            sentStart2 = modifiedCommunicator.numberBitsSent2
            recievedStart = modifiedCommunicator.totalBitsRecieved
            recievedStart0 = modifiedCommunicator.numberBitsRecieved0
            recievedStart1 = modifiedCommunicator.numberBitsRecieved1
            recievedStart2 = modifiedCommunicator.numberBitsRecieved2

            v1, v2 = dotprod(rank, numClients, tl1, tl2, k1, k2)
            v1_, v2_ = dotprod(rank, numClients, tl1_, tl2_, k1_, k2_)
            v1 = v1 ^ v1_
            v2 = v2 ^ v2_
            reconstOutput(rank, [[0]], [[v1]], [[v2]])

            sentEnd = modifiedCommunicator.totalBitsSent
            sentEnd0 = modifiedCommunicator.numberBitsSent0
            sentEnd1 = modifiedCommunicator.numberBitsSent1
            sentEnd2 = modifiedCommunicator.numberBitsSent2
            recievedEnd = modifiedCommunicator.totalBitsRecieved
            recievedEnd0 = modifiedCommunicator.numberBitsRecieved0
            recievedEnd1 = modifiedCommunicator.numberBitsRecieved1
            recievedEnd2 = modifiedCommunicator.numberBitsRecieved2
            modifiedCommunicator.totalBitsSent += (sentEnd - sentStart)*46
            modifiedCommunicator.totalBitsRecieved += (recievedEnd - recievedStart)*46
            modifiedCommunicator.numberBitsSent0 += (sentEnd0 - sentStart0)*46
            modifiedCommunicator.numberBitsRecieved0 += (recievedEnd0 - recievedStart0)*46
            modifiedCommunicator.numberBitsSent1 += (sentEnd1 - sentStart1)*46
            modifiedCommunicator.numberBitsRecieved1 += (recievedEnd1 - recievedStart1)*46
            modifiedCommunicator.numberBitsSent2 += (sentEnd2 - sentStart2)*46
            modifiedCommunicator.numberBitsRecieved2 += (recievedEnd2 - recievedStart2)*46

    if rank == 2:
        c = np.ones((numClients, numBlocks)).astype('uint64')
        t2 = np.array(ta)
        t0 = np.array(tb)
        t2_ = np.array(ta_)
        t0_ = np.array(tb_)
        tl2 = np.sum(t2 & c, axis = 1)%2
        tl0 = np.sum(t0 & c, axis = 1)%2
        k2 = np.array([i[-1]%2 for i in t2])
        k0 = np.array([i[-1]%2 for i in t0])

        tl2_ = np.sum(t2 & c, axis = 1)%2
        tl0_ = np.sum(t0 & c, axis = 1)%2
        k2_ = np.array([i[-1]%2 for i in t2_])
        k0_ = np.array([i[-1]%2 for i in t0_])
        for i in range(2):
            sentStart = modifiedCommunicator.totalBitsSent
            sentStart0 = modifiedCommunicator.numberBitsSent0
            sentStart1 = modifiedCommunicator.numberBitsSent1
            sentStart2 = modifiedCommunicator.numberBitsSent2
            recievedStart = modifiedCommunicator.totalBitsRecieved
            recievedStart0 = modifiedCommunicator.numberBitsRecieved0
            recievedStart1 = modifiedCommunicator.numberBitsRecieved1
            recievedStart2 = modifiedCommunicator.numberBitsRecieved2

            v2, v0 = dotprod(rank, numClients, tl2, tl0, k2, k0)
            v2_, v0_ = dotprod(rank, numClients, tl2_, tl0_, k2_, k0_)

            v2 = v2 ^ v2_
            v0 = v0 ^ v0_
            reconstOutput(rank, [[0]], [[v2]], [[v0]])

            sentEnd = modifiedCommunicator.totalBitsSent
            sentEnd0 = modifiedCommunicator.numberBitsSent0
            sentEnd1 = modifiedCommunicator.numberBitsSent1
            sentEnd2 = modifiedCommunicator.numberBitsSent2
            recievedEnd = modifiedCommunicator.totalBitsRecieved
            recievedEnd0 = modifiedCommunicator.numberBitsRecieved0
            recievedEnd1 = modifiedCommunicator.numberBitsRecieved1
            recievedEnd2 = modifiedCommunicator.numberBitsRecieved2
            modifiedCommunicator.totalBitsSent += (sentEnd - sentStart)*46
            modifiedCommunicator.totalBitsRecieved += (recievedEnd - recievedStart)*46
            modifiedCommunicator.numberBitsSent0 += (sentEnd0 - sentStart0)*46
            modifiedCommunicator.numberBitsRecieved0 += (recievedEnd0 - recievedStart0)*46
            modifiedCommunicator.numberBitsSent1 += (sentEnd1 - sentStart1)*46
            modifiedCommunicator.numberBitsRecieved1 += (recievedEnd1 - recievedStart1)*46
            modifiedCommunicator.numberBitsSent2 += (sentEnd2 - sentStart2)*46
            modifiedCommunicator.numberBitsRecieved2 += (recievedEnd2 - recievedStart2)*46

    v = 0
    return v    
