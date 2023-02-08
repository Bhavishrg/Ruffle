import sys

# sys.path.insert(0, '../../')
# from cryptenLocal.SwiftMPC.ab import ruffleAB
from clarion import *
import argparse

parser = argparse.ArgumentParser(
                    prog = 'Anonymous Broadcast',
                    description = 'Anonymous Broadcast implementation of Clarion and Ruffle')

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--scheme', type=str, required=True, help ="['clarion', 'ruffle']")
parser.add_argument('-n', '--numclients', type=int, required=True, help = "number of clients")
parser.add_argument('-m', '--msgblocks', type=int, required=True, help = "number of blocks of messages of size 8 Bytes")

args = parser.parse_args()
scheme = args.scheme
n = args.numclients
m = args.msgblocks
