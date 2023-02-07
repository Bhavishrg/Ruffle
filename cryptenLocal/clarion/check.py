messageValue = b""

for i in range(0, 2):
    messageValue = messageValue + (97 + 1).to_bytes(1, byteorder = 'big')

print(messageValue)
print(messageValue[0])
print(isinstance(messageValue, (bytes, bytearray)))