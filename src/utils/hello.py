import sys
from time import sleep

for i in range(10000):
    print("\r{} pased".format(i), end='')
    sleep(0.5)