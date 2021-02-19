""" 
    Reference:
    Iwata, Tomoharu, Amar Shah, and Zoubin Ghahramani.
    "Discovering latent influence in online social activities
    via shared cascade poisson processes."
    Proceedings of the 19th ACM SIGKDD international conference
    on Knowledge discovery and data mining. 2013.
"""

import warnings
import numpy as np
from scipy.special import digamma, gammaln
from tqdm import tqdm, trange


class SCPP():
    def __init__(self):

        # Parameters

        self.alpha_i = None
        self.alpha_u = None
        self.theta = None

    def _init_variables(self):

        # Latent variable to keep causality
        self.Z = [np.full(len(D), -1, dtype='int32') for D in self.events]

        # Variable to keep counts of causal users
        self.M = np.zeros((self.n_users + 1, self.n_users))

        # Assume that all events are caused by the back ground intensity
        for ii, Di in enumerate(self.events):
            for u in range(self.n_users):
                self.M[-1, u] += len(Di.query('user_id==@u'))

    def _prep_transaction(self, data):

        data['item_id'] = data['item_id'].astype('int32')
        data['user_id'] = data['user_id'].astype('int32')
        data['date_id'] = data['date_id'].astype('int32')

        # Stats

        self.n_items = data['item_id'].max() + 1
        self.n_users = data['user_id'].max() + 1
        self.T = data['date_id'].max()

        # List of event history per item

        self.events = []
        for i in range(self.n_items):
            self.events.append(
                data.query('item_id==@i').sort_values('date_id').reset_index())

    def fit(self, data, gamma=1, beta=2, a=1, b=1,
            max_iter=100, tol=1e+2, min_gamma=1e-10, max_beta=1e+12,
            verbose=True):

        # Initialization

        self._prep_transaction(data)
        self._init_variables()

        train_hist = []  # history of loglikelihood

        for iteration in range(max_iter):

            # E step

            self.collapsed_gibbs_sampling(beta, gamma, a, b)

            # M step

            gamma = self.update_gamma(gamma, a, b)
            beta  = self.update_beta(beta)

            # Copmute loglikelihood

            # if iteration > 2 and train_hist[-1] - train_hist[-2] < tol:
            #     break

            if verbose == True:
                print()
                print('=' * 20)
                print('Iteration =', iteration + 1)
                print('=' * 20)
                print('gamma =', gamma)
                print('beta  =', beta)
                print()

        # Compute results

        Mi = np.array([(Zi == -1).sum() for Zi in self.Z])  #
        Mu = np.array([self.Mu(u) for u in range(self.n_users)])
        Ci = self.T
        Cu = np.array([self.Cu(Ci, gamma, u) for u in range(self.n_users)])

        self.alpha_i = (Mi + a) / (Ci + b)
        self.alpha_u = (Mu + a) / (Cu + b)

        self.theta = np.array([
            (self.M[u] + beta)
            / (self.Mu(u) + beta * self.n_users)
            for u in range(self.n_users)
        ])

    def collapsed_gibbs_sampling(self, beta, gamma, a, b):

        # desc = 'CollapsedGibbsSampling'
        for ii, Di in enumerate(self.events):

            desc = 'Item {}'.format(ii + 1)

            for ei, Dit in tqdm(Di.iterrows(), total=len(Di), desc=desc):
                # print(ei, Dit)
                # Draw an event
                t, u = Dit[['date_id', 'user_id']]
                # print(Dit)
                # print(t, u)

                if t == 0:
                    # events caused by background intensity
                    self.Z[ii][ei] = -1
                    continue
                
                # Reset z_in
                
                z_prev = self.Z[ii][ei]
                u_prev = -1 if z_prev == -1 else Di.iloc[z_prev]['user_id']
                self.M[u_prev, u] -= 1

                # Compute a distribution to draw a causal event

                Dy = Di.query('date_id<@t')
                ny = len(Dy)
                pz = np.zeros(ny + 1)

                # of events caused by the back ground intensity in item i
                Mi = len(self.Z[ii][self.Z[ii] == -1])
                Ci = self.T

                pz[0] = (Mi + a) * (self.M[-1, u] + beta)
                pz[0] /= (Ci + b) * (self.M[-1].sum() + beta * self.n_users)

                for y, Dyi in Dy.iterrows():

                    ty, uy = Dyi[['date_id', 'user_id']]
                    num = (self.M[uy].sum() + a) * (self.M[uy, u] + beta)
                    den = (self.Cu(self.T, gamma, uy) + b) * (self.M[uy].sum() + beta * self.n_users)
                    pz[y + 1] = np.exp(-1 * gamma * (t - ty)) * num / den

                pz = pz / pz.sum()  # normalize [0 1]

                z = np.random.choice(np.arange(ny + 1, dtype=int), size=1, p=pz) - 1
                self.Z[ii][ei] = z

                uz = -1 if z == -1 else Di.iloc[z]['user_id']
                self.M[uz, u] += 1

    def Mu(self, u):
        # u: -1:n_users-1
        return self.M[u].sum()
    
    def Cu(self, T, gamma, u):
        C = 0

        for i, Di in enumerate(self.events):
            t = Di.query('user_id==@u')['date_id'].values
            C += (1 - np.exp(-1 * gamma * (T - t))).sum()

        return C / gamma

    def update_gamma(self, gamma, a, b):
        """ Newton's method (Equation 29 and 30)
        """
        de1 = de2 = 0

        for ii, (Di, Zi) in enumerate(zip(self.events, self.Z)):

            idx = Zi > -1
            Dit = Di.iloc[idx]['date_id'].values
            Zit = Di.iloc[Zi[idx]]['date_id'].values
            de1 -= (Dit - Zit).sum()

        for u in range(self.n_users):

            Au = self.compute_A(u, gamma)
            # Mu = self.Mu(u)
            Mu = self.M[u].sum(0)

            val1 = (Mu + a) / (Au[0] + gamma * b)
            val2 = -1 * Au[0] / gamma + Au[1]

            de1 -= val1 * val2
            de2 -= val1 * (Au[2] + val2 * (
                (Au[1] + b) / (Au[0] + gamma + b) + 1 / gamma))

        return max(0, gamma - de1 / de2)  # Equation (18)

    def compute_A(self, u, gamma):
        """ Subfunction for update_gamma
        """

        A0 = gamma * self.Cu(self.T, gamma, u)
        A1 = A2 = 0
        
        for ii, Di in enumerate(self.events):

            t = Di.query('user_id==@u')['date_id'].values  # array

            diff = self.T - t
            exp_mgamma_diff = np.exp(-1 * gamma * diff)

            A1 += (exp_mgamma_diff * diff).sum()
            A2 += (exp_mgamma_diff * np.power(diff, 2)).sum()

        return A0, A1, A2

    def update_beta(self, beta, minimum=1e-10):
        """ Fixed-point iteration method """

        # U+ means all practical users
        # and the virtual user for background intensity
        # digamma(0) = -inf
        U = self.n_users
        num = (digamma(self.M + beta) - digamma(beta)).sum()
        # den = (digamma(self.M.sum(axis=1) + beta * U) - digamma(beta * U)).sum()

        # https://tminka.github.io/papers/dirichlet/minka-dirichlet.pdf
        # Equation (55)
        # num = digamma(self.M + beta).sum() - digamma(beta)
        den = digamma(self.M.sum(axis=1) + beta * U).sum() - digamma(beta * U)
        # print()
        # print('num', num)
        # print('den', den)

        # num = -1 * (self.n_users + 1) * self.n_users * digamma(beta)
        # den = -1 * (self.n_users + 1) * digamma(beta * self.n_users)
        # print()
        # print('num', num)
        # print('den', den)

        # num += digamma(self.M + beta).sum()
        # den += digamma(self.M.sum(axis=1) + beta * self.n_users).sum()
        # print()
        # print('num', num)
        # print('den', den)

        # for u in trange(-1, self.n_users, desc='UpdateBeta'):
        #     num += sum([digamma(self.M[u, v] + beta) for v in range(self.n_users)])
        #     den += digamma(self.Mu(-1) + beta * self.n_users)
        
        return max(minimum, beta * num / den)  # Equation (19)

    def loglikelihood(self, gamma, beta, a, b):
        # Joint log likelihood

        # for ii, Di in enumerate(self.events):
        #     for 
        pass

    def inference():
        pass

    def save(self, fp, save_params_only=True):
        """
        """

        if save_params_only == True:
            np.savetxt(fp + 'alpha_i.txt', self.alpha_i)
            np.savetxt(fp + 'alpha_u.txt', self.alpha_i)
            np.savetxt(fp + 'theta.txt', self.theta)