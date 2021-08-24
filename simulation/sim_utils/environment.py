import numpy as np
from collections import namedtuple


def sample_cov_matrix(d, k, random_state, sd=1.):
    W = random_state.randn(d, k)
    S = W.dot(W.T) + np.diag(random_state.rand(d))
    inv_std = np.diag(1. / np.sqrt(np.diag(S)))
    corr = inv_std.dot(S).dot(inv_std)
    diag_std = np.eye(d) * np.repeat(sd, d)
    cov = diag_std.dot(corr).dot(diag_std)
    return cov


class Environment(object):

    def __init__(self, n_env, n_actions, inv_seed, non_inv_seed, train=True):
        self.n_env = n_env
        self.n_actions = n_actions
        inv_state = np.random.RandomState(inv_seed)
        self.bH3 = inv_state.normal(0, 1)
        self.bR = inv_state.normal(0, 1, size=(n_actions, 2))
        non_inv_state = np.random.RandomState(non_inv_seed)
        if train:
            scale = 2
            self.bH1 = non_inv_state.normal(size=(self.n_env, 1), scale=scale)
            self.bH2 = non_inv_state.normal(size=(self.n_env, 1), scale=scale)
        else:
            scale = 4
            self.bH1 = non_inv_state.normal(size=(self.n_env, 1), scale=scale)
            self.bH2 = non_inv_state.normal(size=(self.n_env, 1), scale=scale)

        self.e = np.stack([self.bH1, self.bH2])

    def gen_data(self, train_policy, s_size):
        # gendata
        D = []
        for env in range(self.n_env):
            H1, H2, H3 = get_state(s_size, self.bH3, bH1=self.bH1[env, :], bH2=self.bH2[env, :])
            # visible state
            X = np.stack([H1, H2], axis=1)
            # choose actions
            A, P = train_policy.get_actions(X)
            R = reward(H2, H3, self.bR)

            E = np.repeat(env, R.shape[0])
            D += [(X, A, R, P, E)]

        X, A, R, P, E = [np.stack(arr).reshape(s_size * self.n_env, -1) for arr in list(zip(*D))]
        A, P, E = A.squeeze(), P.squeeze(), E.squeeze()

        return X, A, R, P, E

    def get_corr(self):
        s_size = 10000
        H3_vec = []
        H1_vec = []
        for env in range(self.n_env):
            H1, H2, H3 = get_state(s_size, self.bH3, bH1=self.bH1[env, :], bH2=self.bH2[env, :])
            H3_vec += [H3]
            H1_vec += [H1]
        corr = np.array([np.corrcoef(h3, h1)[1, 0] for h3,h1 in zip(H3_vec, H1_vec)])
        # H3_vec = np.concatenate(H3_vec)
        # H1_vec = np.concatenate(H1_vec)

        return corr


def get_state(n, bH3, bH1, bH2):
    H3 = np.random.normal(bH3, size=(n,))
    H2 = np.random.normal(bH2, size=(n,))
    H1 = np.random.normal(bH1 * H3)

    return H1, H2, H3


def reward(H2, H3, bR):
    # compute reward
    return bR.dot(np.stack([H2, H3])).T
