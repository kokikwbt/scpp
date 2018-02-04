import numpy as np
import scipy.stats as st
import scipy.special as sp
from progressbar import ProgressBar

from synthetic import *
ZERO = 1.e-10
INF = 1.e+10
MAX_ITER = 1000

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
    return len(np.where(Z[item_id]==0)[0])

def M_uu(user1, user2):
    cnt = 0
    for i in range(n_items):
        cause = np.where(Z[i]==user1)[0]
        effect = np.where(D[i][cause, 0]==user2)[0]
        cnt += len(effect)
    return cnt

def M_u(user):
    return sum([len(np.where(Z[i]==user)[0]) for i in range(n_items)])

def A_u0(gamma, user):
    return gamma * C_u(gamma, user)
# A'
def A_u1(gamma, user):
    val = 0.
    for i in range(n_items):
        T = D[i][-1, 0]
        idx = np.where(D[i][:, 1] == user)[0]
        t = D[i][idx, 0] #.astype(np.float64)
        val += np.sum(np.exp(-gamma * (T - t)) * (T - t))
    return -1 * val
# A''
def A_u2(gamma, user):
    val = 0.
    for i in range(n_items):
        idx = np.where(D[i][:, 1] == user)[0]
        t = D[i][idx, 0] #.astype(np.float64)
        T = D[i][-1, 0]
        val += np.sum(np.exp(-gamma * (T - t)) * np.power((T - t), 2))
    return val

def update_gamma(gamma, a, b):
    deriv1 = deriv2 = 0.
    for i in range(n_items):
        idx = np.where(Z[i] > 0)[0]
        deriv1 += np.sum(D[i][idx][:, 0] - D[i][Z[i][idx]][:, 0])
    for user in range(1, n_users):
        # compute components
        M = M_u(user)
        Au0 = A_u0(gamma, user)
        Au1 = A_u1(gamma, user)
        Au2 = A_u2(gamma, user)
        val1 = (M + a) / (Au0 + np.power(gamma, b))
        val2 = -1 * Au1 / gamma + Au1
        # for first derivative
        deriv1 += val1 * val2
        # for second derivative
        deriv2 += val1 * (Au2 + val2 * ((Au1 + b) / (Au0 + np.power(gamma, b)) + 1 / gamma))
    deriv1 *= -1
    deriv2 *= -1
    return gamma - deriv1 / deriv2

def update_beta(beta):
    term1 = term2 = 0.
    for u0 in range(n_users):
        for u1 in range(1, n_users):
            term1 += sp.digamma(M_uu(u0, u1) + beta) - sp.digamma(beta)
    for u in range(n_users):
        term2 += sp.digamma(M_u(u) + beta * n_users) - sp.digamma(beta * n_users)
    return beta * term1 / term2

def compute_lh(gamma, beta, a, b):
    # the joint likelihood p(D, Z| gamma, beta, a, b)
    llh = 0.
    cnt = 0
    for i in range(n_items):
        events = D[i]
        for n in range(1, len(events)):
            zin = Z[i][n]
            if zin == 0: continue
            cnt += 1
            llh += events[n, 0] - events[zin, 0]
    print('# of cascade:', cnt)
    llh = -gamma * llh
    llh += (n_users + n_items) * np.log(np.power(b, a) / sp.gamma(a))
    for u in range(1, n_users):
        Mu = M_u(u)
        llh += np.log(sp.gamma(Mu + a) / np.power(C_u(gamma, u) + b, Mu + a))
    for i in range(n_items):
        Mi = M_i(i)
        llh += np.log(sp.gamma(Mi + a) / np.power(D[i][-1, 0] + b, Mi + a))
    llh += (n_users + 1) * np.log(sp.gamma(beta * n_users) / np.power(sp.gamma(beta), n_users))
    for u0 in range(n_users):
        for u1 in range(1, n_users):
            llh += np.log(sp.gamma(M_uu(u0, u1) + beta))
        llh -= np.log(sp.gamma(M_u(u0) + beta * n_users))
    return llh

def sample_latent_index(gamma, beta, a, b):
    progress = ProgressBar(0, n_items).start()
    for item_id in range(n_items):
        progress.update(item_id)
        T = D[item_id][-1, 0] # Ci = T
        for eid in range(len(D[item_id])):
            if eid == 0:
                Z[item_id][0] = 0
                continue
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
            for i in range(1, eid):
                tiy, uiy = D[item_id][i]
                pz[i] = np.exp(-gamma * (tin - tiy))
                pz[i] *= M_u(uiy) + a
                pz[i] /= C_u(gamma, uiy) + b
                pz[i] *= M_uu(uiy, uin) + beta
                pz[i] /= M_u(uiy) + beta * n_users
            pz = pz / np.sum(pz)
            Z[item_id][eid] = np.random.choice(np.arange(eid), size=1, p=pz)
    print('')

def inference(gamma=10, beta=0.1, a=1, b=1):
    # stochastic EM algorithm
    lh = prev = 0.
    for i in range(MAX_ITER):
        print('===============')
        print('Iter: ', i + 1)
        print('===============')
        print('E-step:')
        print('sampling latent indexes...')
        # E step
        sample_latent_index(gamma, beta, a, b)
        # M step
        print('M-step:')
        prev = gamma
        gamma = update_gamma(gamma, a, b)
        print('Gamma: ', prev, '->', gamma)
        if gamma < 0: gamma = ZERO
        prev = beta
        beta = update_beta(beta)
        print('Beta: ', prev, '->', beta)
        # if not i % 5:
        prev = lh
        lh = compute_lh(gamma, beta, a, b)
        print('Likelihood =', prev)
        print('Likelihood =', lh)
            # if np.fabs(lh - prev) < 10: break


if __name__ == '__main__':

    n_items = 500
    n_users = 20 + 1 # of users + background
    # marked point process
    D = import_data(n_items)
    # latent variable set
    Z = [np.zeros(len(D[i]), dtype=np.int64) for i in range(n_items)]

    inference()
