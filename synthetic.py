import os, shutil, sys
from numpy import int64
from numpy import zeros
from numpy import savetxt
from numpy import loadtxt
from numpy.random import *
import matplotlib.pyplot as plt

def marked_point_process(n_users):
    n_events = randint(5, 10)
    D = zeros((n_events, 2), dtype=int64)
    for i in range(n_events):
        user = -1
        if i > 1:
            if D[i - 1, 1] == 3:
                user = choice([1, 4, 9], p=[0.2, 0.7, 0.1])
            elif D[i - 1, 1] == 20:
                user = 8
            else:
                user = randint(n_users)
        else:
            user = randint(n_users)
        print(user)
        D[i, 0] = exponential(200) + D[i - 1, 0]# time
        D[i, 1] = user + 1 # user ID
    print(D)
    return D

def import_data(n_items):
    return [loadtxt('./synthetic/item'+str(i)) for i in range(n_items)]

if __name__ == '__main__':
    if not len(sys.argv) == 3:
        exit('arg error')
    n_items = sys.argv[1]
    n_users = sys.argv[2]
    outdir = './synthetic_' + n_items + '_' + n_users + '/'
    if os.path.exists(outdir):
        shutil.rmtree(outdir)
    os.mkdir(outdir)
    for i in range(int(n_items)):
        D = marked_point_process(int(n_users))
        savetxt(outdir + 'item' + str(i), D)
