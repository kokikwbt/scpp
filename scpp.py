import numpy as np
from synthetic import *

global D, Z
global n_items
global n_users


def C_u(gamma, user):
    C = 0.
    for i in range(n_items):
        T = D[i][-1, 0]
        idx = np.where(D[i][:, 1] == user)[0]
        t = D[i][idx, 0] #.astype(np.float64)
        C += np.sum(1 - np.exp(-gamma * (T - t))) / gamma
    return C

def M_i(item_id):
    zin = Z[item_id]
    return len(np.where(zin==0)[0])

def M_uu(user1, user2):
    # ignore item -1, event -1
    cnt = 0
    for i in range(n_items):
        zin = Z[i]
        cause = np.where(zin==user1)[0]
        events = D[i][cause, 0]
        effect = np.where(events==user2)[0]
        cnt += len(effect)
    return cnt

def M_u(user):
    # ignore item -1, event -1
    cnt = 0
    for i in range(n_items):
        zin = Z[i]
        cnt += len(np.where(zin==user)[0])
    return cnt

def A_u3(gamma, user):
    val = 0.
    for i in range(n_items):
        T = D[i][-1, 0]
        idx = np.where(D[i][:, 1] == user)[0]
        t = D[i][idx, 0] #.astype(np.float64)
        val += np.exp(-gamma * (T - t)) * np.power((T - t), 2)
    return val

def A_u2(gamma, user):
    val = 0.
    for i in range(n_items):
        T = D[i][-1, 0]
        idx = np.where(D[i][:, 1] == user)[0]
        t = D[i][idx, 0] #.astype(np.float64)
        val += np.exp(-gamma * (T - t)) * (T - t)
    return -val

def A_u(gamma, user):
    return gamma * C_u(user)

def update_gamma(gamma, a, b):
    val1 = val2 = val3 = 0.
    for i in range(n_items):
        idx = np.where(Z[i] > 0)[0]
        val1 += np.sum(D[i][idx][:, 0] - D[i][Z[i][idx]][:, 0])
    for user in range(n_users):
        # compute components
        M = M_u(user)
        Au = A_u(gamma, user)
        Aup = A_u2(gamma, user)
        Aupp = A_u3(gamma, user)
        # for first derivative
        tmp = M + a
        tmp /= Au + gamma * b
        hoge = -Au / gamma + Aup
        tmp *= hoge
        val2 += tmp
        # for second derivative
        tmp = M + a
        tmp /= Au + gamma * b
        tmp *= Aupp + hoge * ((Aup + b) / (Au + gamma * b) + 1 / gamma)
        val3 += tmp
    return gamma - (-(val1+val2)) / val3

def compute_lh():
    return 0.

def E_step(gamma, beta, a, b):
    for item_id in range(len(D)):
        T = D[item_id][-1, 0]
        for eid in range(1, len(D[item_id])): # zin of the first events is zero?
            # reset z_in
            Z[item_id][eid] = -1
            pz = np.zeros(eid)
            tin, uin = D[item_id][eid]
            # background intensity
            pz[0] = M_i(item_id) + a 
            pz[0] /= T + b
            pz[0] *= M_uu(0, uin) + beta
            pz[0] /= M_u(0) + beta * n_users
            # otherwise
            for i in range(1, eid - 1):
                tiy, uiy = D[item_id][i]
                pz[i] = np.exp(-gamma * (tin - tiy))
                pz[i] *= M_u(uiy) + a
                pz[i] /= C_u(gamma, uiy) + b
                pz[i] *= M_uu(uiy, uin) + beta
                pz[i] /= M_u(uiy) + beta * n_users
            denom = np.sum(pz)
            pz = pz / denom
            # sample latent index
            Z[item_id][eid] = np.random.choice(np.arange(eid), size=1, p=pz)
            print(Z[item_id][eid])

def M_step():
    # update Gamma

    # update Beta

    pass


def inference(gamma=0.01, beta=1, a=1, b=1):
    # stochastic EM algorithm
    lh = prev = 0.
    for i in range(10000):
        print('------')
        print('E-step')
        print('------')
        E_step(gamma=gamma, beta=beta, a=a, b=b)

        # M step
        print(gamma)
        gamma = update_gamma(gamma, a, b)
        print('->', gamma)
        exit()
        # beta = update_beta()

        if not i % 5:
            lh = compute_lh()
            print('Likelihood=', lh)
            if np.fabs(lh - prev) < 10: break


if __name__ == '__main__':

    n_items = 100
    n_users = 100
    # marked point process
    D = import_data(n_items)
    # latent variable set
    Z = [np.zeros(len(D[i]), dtype=np.int64) - 1 for i in range(n_items)]

    inference()
