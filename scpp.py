import sys

import numpy as np
import scipy.stats as st
import scipy.special as sp
from progressbar import ProgressBar

from synthetic import *

OUTDIR = './result/dat_tmp/'
ZERO = 1.e-10
INF = 1.e+10
MAX_ITER = 1000
TH = 1.e+1

global D, Z, T, Mu, Muu
global Cu
global n_items
global n_users

class Workspace:
    def __init__(self, n_items, n_users):
        # init latent variable set
        self.Z = [np.zeros(len(D[i]), dtype=np.int64) - 1 for i in range(n_items)]
        # Z = [np.arange(len(D[i]), dtype=np.int64) - 1 for i in range(n_items)]

        # set observation period
        self.T = max([D[i][-1, 0] for i in range(n_items)])

        # init count
        self.M = np.zeros((n_users, n_users))
        self.M_ = np.zeros(n_users) # for bg
        for i in range(n_items):
            D_ = D[i][:, 1]
            for u in range(n_users):
                self.M_[u] += len(D_[D_ == u])

    def Mu(self, user_id):
        if user_id == -1:
            return np.sum(self.M_)
        else:
            return np.sum(self.M[user_id, :])


def C_u(T, gamma, user):
    C = 0.
    for i in range(n_items):
        idx = np.where(D[i][:, 1] == user)[0]
        if len(idx) == 0: continue
        t = D[i][idx, 0] #.astype(np.float64)
        C += np.sum(1 - np.exp(-gamma * (ws.T - t))) / gamma
    return C

def A_u0(T, gamma, user):
    return gamma * C_u(T, gamma, user)

# A'
def A_u1(T, gamma, user):
    val = 0.
    for i in range(n_items):
        idx = np.where(D[i][:, 1] == user)[0]
        t = D[i][idx, 0] #.astype(np.float64)
        val += np.sum(np.exp(-gamma * (T - t)) * (T - t))
    return -1 * val
# A''
def A_u2(T, gamma, user):
    val = 0.
    for i in range(n_items):
        idx = np.where(D[i][:, 1] == user)[0]
        t = D[i][idx, 0] #.astype(np.float64)
        val += np.sum(np.exp(-gamma * (T - t)) * np.power((T - t), 2))
    return val

def update_gamma(ws, gamma, a, b):
    print('\nupdate gamma...')
    deriv1 = deriv2 = 0.
    for i in range(n_items):
        idx = np.where(ws.Z[i] > -1)[0]
        deriv1 += np.sum(D[i][idx, 0] - D[i][ws.Z[i][idx], 0])

    progress = ProgressBar(0, n_users).start()
    for user in range(n_users):
        progress.update(user)
        # compute components
        Au0 = A_u0(ws.T, gamma, user)
        Au1 = A_u1(ws.T, gamma, user)
        Au2 = A_u2(ws.T, gamma, user)
        val1 = (ws.Mu(user) + a) / (Au0 + gamma * b)
        val2 = -1 * Au0 / gamma + Au1
        # for first derivative
        deriv1 += val1 * val2
        # for second derivative
        deriv2 += val1 * (Au2 + val2 * ((Au1 + b) / (Au0 + gamma * b) + 1 / gamma))
    return gamma - deriv1 / deriv2 if gamma - deriv1 / deriv2 > 0 else ZERO

def update_beta(ws, beta):
    print('\nupdate beta...')
    progress = ProgressBar(0, n_users + 1).start()
    # for background intensity
    term1 = np.sum([sp.digamma(ws.M_[u1] + beta * n_users) - sp.digamma(beta * n_users) for u1 in range(n_users)])
    term2 = sp.digamma(ws.Mu(-1) + beta) - sp.digamma(beta)

    for u0 in range(n_users):
        progress.update(u0)
        Mu = ws.Mu(u0)
        if Mu == 0: continue
        term1 += np.sum([sp.digamma(ws.M[u0, u1] + beta) - sp.digamma(beta) for u1 in range(n_users)])
        term2 += sp.digamma(Mu + beta) - sp.digamma(beta)
    return beta * term1 / term2

def compute_lh(ws, gamma, beta, a, b):
    # the joint likelihood p(D, Z| gamma, beta, a, b)
    llh = cnt = 0
    for i in range(n_items):
        for n in range(len(D[i])):
            zin = ws.Z[i][n]
            if zin == -1: continue
            llh += D[i][n, 0] - D[i][zin, 0]
            cnt += 1
    print('\n# of cascade:', cnt)
    llh = -1 * gamma * llh

    llh += (n_users + n_items) * np.log(np.power(b, a) / sp.gamma(a))
    for u in range(n_users):
        Mu = ws.Mu(u)
        llh += np.log(sp.gamma(Mu + a))
        llh -= (Mu + a) * np.log(C_u(ws.T, gamma, u) + b)
    for i in range(n_items):
        Mi = len(ws.Z[i][ws.Z[i] == -1])
        llh += np.log(sp.gamma(Mi + a))
        llh -= (Mi + a) * np.log(ws.T + b)

    llh += (n_users + 1) * (sp.gammaln(beta * n_users) - n_users * sp.gammaln(beta))
    llh += np.sum([sp.gammaln(ws.M_[u1] + beta) for u1 in range(n_users)])
    llh -= sp.gammaln(ws.Mu(-1) + beta * n_users)
    for u0 in range(n_users):
        llh += np.sum([sp.gammaln(ws.M[u0, u1] + beta) for u1 in range(n_users)])
        llh -= sp.gammaln(ws.Mu(u0) + beta * n_users)
    return llh

def sample_latent_index(ws, gamma, beta, a, b):
    print('sampling latent indexes...')
    progress = ProgressBar(0, n_items).start()
    for item_id in range(n_items):
        progress.update(item_id)
        D_ = D[item_id]
        for eid in range(len(D_)):
            if eid == 0:
                ws.Z[item_id][0] = -1
                continue
            tin, uin = D_[eid]
            tin, uin = int(tin), int(uin)

            # reset z_in
            z_old = ws.Z[item_id][eid]
            if z_old == -1:
                ws.M_[uin] -= 1 # decrement
            else:
                _, u_old = D_[z_old]
                ws.M[int(u_old), uin] -= 1 # decrement

            ws.Z[item_id][eid] = -INF
            pz = np.zeros(eid + 1)
            # if y = 0
            Mi = len(ws.Z[item_id][ws.Z[item_id] == -1])
            pz[0] = Mi + a 
            pz[0] /= ws.T + b
            pz[0] *= ws.M_[uin] + beta
            pz[0] /= ws.Mu(-1) + beta * n_users
            # otherwise
            for y in range(eid):
                tiy, uiy = D_[y]
                tiy, uiy = int(tiy), int(uiy)
                pz[y + 1] = np.exp(-1 * gamma * (tin - tiy))
                pz[y + 1] *= ws.Mu(uiy) + a
                pz[y + 1] /= C_u(ws.T, gamma, uiy) + b
                pz[y + 1] *= ws.M[uiy, uin] + beta
                pz[y + 1] /= ws.Mu(uiy) + beta * n_users
            pz[pz < ZERO] = ZERO
            pz = pz / np.sum(pz)
            cause = np.random.choice(np.arange(eid + 1), size=1, p=pz) - 1
            ws.Z[item_id][eid] = cause
            if not cause == -1:
                uiz = int(D_[cause, 1])
                ws.M[uiz, uin] += 1
            else:
                ws.M_[uin] += 1

def inference(ws, gamma=1, beta=2, a=1, b=1):
    # stochastic EM algorithm
    prev = -INF
    buff = 0
    for i in range(MAX_ITER):
        print('============')
        print('Iter: ', i + 1)
        print('============')

        # E step
        sample_latent_index(ws, gamma, beta, a, b)

        # M step
        gamma = update_gamma(ws, gamma, a, b)
        if gamma < 0:
            exit('invalid gamma')
        beta = update_beta(ws, beta)

        lh = compute_lh(ws, gamma, beta, a, b)
        print('\n\nGamma =', gamma)
        print('\n\nBeta  =', beta)
        diff = lh - prev
        if diff > 0:
            print('\nL-likelihood = {0} (+{1})\n'.format(lh, lh - prev))
        else:
            print('\nL-likelihood = {0} ({1})\n'.format(lh, lh - prev))
        prev = lh
        if np.fabs(diff) < TH:
            buff += 1
            if buff > 10:
                break

    Mi = np.array([len(ws.Z[i][ws.Z[i] == -1]) for i in range(n_items)])
    Mu = np.array([ws.Mu(u) for u in range(n_users)])
    Ci = ws.T
    Cu = np.array([C_u(ws.T, gamma, u) for u in range(n_users)])

    alpha_i = (Mi + a) / (Ci + b)
    alpha_u = (Mu + a) / (Cu + b)
    theta_uu = np.zeros((n_users, n_users))
    for u in range(n_users):
        theta_uu[u] = (ws.M[u] + beta) / (Mu[u] + beta * n_users)

    if os.path.exists(OUTDIR):
        shutil.rmtree(OUTDIR)
    os.mkdir(OUTDIR)
    with open(OUTDIR + 'gamma', 'r') as f:
        f.write(gamma)
    with open(OUTDIR + 'beta', 'r') as f:
        f.write(beta)
    np.savetxt(OUTDIR + 'alpha_i', alpha_i)
    np.savetxt(OUTDIR + 'alpha_u', alpha_u)
    np.savetxt(OUTDIR + 'theta_uu', theta_uu)


if __name__ == '__main__':

    # load marked point process
    n_items, n_users = int(sys.argv[1]), int(sys.argv[2])
    D = import_data(n_items, n_users)

    ws = Workspace(n_items, n_users)

    inference(ws)
