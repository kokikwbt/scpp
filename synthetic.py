import os, shutil, sys
from numpy import int64
from numpy import zeros
from numpy import savetxt
from numpy import loadtxt
from numpy.random import *
import matplotlib.pyplot as plt

def marked_point_process(n_users):
    n_events = randint(10, 50)
    D, time = zeros((n_events, 2), dtype=int64), 0
    for i in range(n_events):
        D[i, 0] = exponential(randint(30, 200)) + D[i - 1, 0]# time
        D[i, 1] = randint(n_users) # user ID
    return D

def import_data(n_items):
    return [loadtxt('./synthetic/item'+str(i)) for i in range(n_items)]

if __name__ == '__main__':
    if not len(sys.argv) == 3:
        exit('arg error')
    n_items = int(sys.argv[1])
    n_users = int(sys.argv[2])
    if os.path.exists('./synthetic'):
        shutil.rmtree('./synthetic')
    os.mkdir('./synthetic')
    for i in range(n_items):
        D = marked_point_process(n_users)
        savetxt('./synthetic/item' + str(i), D)