from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.linear_model import Ridge


class RandomForestPolicy(object):

    @property
    def name(self):
        return "rf_learner"

    def __init__(self, subset_idx, n_actions, max_depth=10):
        self.model = {a: RandomForestRegressor(max_depth=max_depth, random_state=0) for a in
                      range(n_actions)}
        self.subset_idx = subset_idx
        self.n_actions = n_actions

    def train(self, X, A, Y, init_pi=None):

        for a in range(self.n_actions):
            if init_pi is not None:
                sample_weight_a = 1 / init_pi[A == a]
                sample_weight_a = sample_weight_a / sample_weight_a.sum()
            else:
                sample_weight_a = None

            # select data corresponding to the action a
            X_a, Y_a = X[:, self.subset_idx][A == a], Y[A == a]
            self.model[a] = self.model[a].fit(X_a, Y_a, sample_weight=sample_weight_a)

    def get_actions(self, X, return_reward=False):
        pred_actions = np.empty(shape=(X.shape[0], self.n_actions), dtype=np.float32)
        for a in range(self.n_actions):
            pred_actions[:, a] = self.model[a].predict(X[:, self.subset_idx])
        if return_reward:
            return np.argmax(pred_actions, axis=1), np.max(pred_actions, axis=1).mean()
        else:
            return np.argmax(pred_actions, axis=1)

    def evaluate(self, X, A, Y):
        errors = []
        for a in range(self.n_actions):
            # select data corresponding to the action a
            X_a, Y_a = X[:, self.subset_idx][A == a], Y[A == a]
            pred_a = self.model[a].predict(X_a)
            errors += [(Y_a - pred_a) ** 2]

        mean_err = np.concatenate(errors).mean()

        return mean_err


class LinearPolicy(object):

    @property
    def name(self):
        return "lr_learner"

    def __init__(self, subset_idx, n_actions, max_depth=10):
        # self.model = {a: Ridge() for a in range(n_actions)}
        self.model = {a: Ridge() for a in range(n_actions)}
        self.subset_idx = subset_idx
        self.n_actions = n_actions

    def train(self, X, A, Y, init_pi=None):

        for a in range(self.n_actions):
            if init_pi is not None:
                sample_weight_a = 1 / init_pi[A == a]
            else:
                sample_weight_a = None

            # select data corresponding to the action a
            X_a, Y_a = X[:, self.subset_idx][A == a], Y[A == a]
            self.model[a] = self.model[a].fit(X_a, Y_a, sample_weight=sample_weight_a)

    def get_actions(self, X, return_reward=False):
        pred_actions = np.empty(shape=(X.shape[0], self.n_actions), dtype=np.float32)
        for a in range(self.n_actions):
            pred_actions[:, a] = self.model[a].predict(X[:, self.subset_idx])
        if return_reward:
            return np.argmax(pred_actions, axis=1), np.max(pred_actions, axis=1).mean()
        else:
            return np.argmax(pred_actions, axis=1)

    def evaluate(self, X, A, Y):
        errors = []
        for a in range(self.n_actions):
            # select data corresponding to the action a
            X_a, Y_a = X[:, self.subset_idx][A == a], Y[A == a]
            pred_a = self.model[a].predict(X_a)
            errors += [(Y_a - pred_a) ** 2]

        mean_err = np.concatenate(errors).mean()

        return mean_err
