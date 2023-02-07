from ab.main import *
from ab.benny import *
from ab.swiftShuffle import *

# benny shuffle for varying table size
# print("starting benny(4,10)")
# benny(4,10)
print("starting benny(4,100)")
benny(4,100)
print("starting benny(4,1000)")
benny(4,1000)
print("starting benny(4,10000)")
benny(4,10000)
# print("starting benny(4,100000)")
# benny(4,100000)
# print("starting benny(4,1000000)")
# benny(4,1000000)


# #swiftshuffle
print("starting swiftShuffle(4, 100)")
swiftShuffle(2, 100)
print("starting swiftShuffle(4, 1000)")
swiftShuffle(2, 1000)
print("starting swiftShuffle(4, 10000)")
swiftShuffle(2, 10000)
# print("starting swiftShuffle(4, 100000)")
# swiftShuffle(4, 100000)
# print("starting swiftShuffle(4, 1000000)")
# swiftShuffle(4, 1000000)


# # ab for varying msg size
print("starting ab(4, 1000)")
ab(4, 1000)
print("starting ab(20, 1000)")
ab(20, 1000)
print("starting ab(125, 1000)")
ab(125, 1000)

# # ab for varying table size
print("starting ab(4, 100)")
ab(4, 100)
print("starting ab(4, 1000)")
ab(4, 1000)
print("starting ab(4, 10000)")
ab(4, 10000)
# print("starting ab(4, 100000)")
# ab(4, 100000)
# print("starting ab(4, 1000000)")
# ab(4, 1000000)


