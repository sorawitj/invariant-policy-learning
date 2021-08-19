import statsmodels.api as sm
import patsy
import numpy as np

from sklearn.ensemble import RandomForestRegressor

def train_lsq(X, R, subset, W=1.):
    params = []
    if R.ndim == 1:
        R = R[:, np.newaxis]
    for a in range(R.shape[1]):
        mod_wls = sm.WLS(R[:, a], sm.tools.add_constant(X[:, subset]), W)
        res_wls = mod_wls.fit()
        params += [res_wls.params]
    params = np.stack(params)
    return params


def predict(X, S, eR):
    return eR.dot(sm.tools.add_constant(X[:, S], has_constant='add').T)


class Learner(object):

    def __init__(self, feature_set):
        self.feature_set = feature_set


class IPWLearner(Learner):
    @property
    def name(self):
        return "ipw_learner"

    def __init__(self, feature_set, n_actions):
        super().__init__(feature_set)
        self.feature_set = feature_set
        self.n_actions = n_actions
        self.f = 'R ~ '
        for s in feature_set:
            self.f += 'X{} * A +'.format(s + 1)
        self.f = self.f[:-1]

    def train(self, df):
        y, X = patsy.dmatrices(self.f, df, return_type='matrix')
        self.wls_model = sm.WLS(y, X, weights=df['w']).fit()

    def eval(self, df, R):
        pred_ys = np.empty(shape=(df.shape[0], self.n_actions), dtype=np.float32)
        for i in range(self.n_actions):
            f = self.f.replace("A", "A{}".format(i))
            _, X = patsy.dmatrices(f, df, return_type='matrix')
            pred_ys[:, i] = self.wls_model.predict(X)

        return np.mean(df['R'] - R[np.arange(R.shape[0]), pred_ys.argmax(1)])

    def pred_reward(self, df):
        pred_ys = np.empty(shape=(df.shape[0], self.n_actions), dtype=np.float32)
        for i in range(self.n_actions):
            f = self.f.replace("A", "A{}".format(i))
            _, X = patsy.dmatrices(f, df, return_type='matrix')
            pred_ys[:, i] = self.wls_model.predict(X)

        return np.max(pred_ys, axis=1).mean()


class RandomForestLearner(Learner):

    @property
    def name(self):
        return "rf_learner"

    def __init__(self, subset_idx, n_actions, max_depth=15):
        super().__init__(subset_idx)
        self.model = {a: RandomForestRegressor(max_depth=max_depth, random_state=0) for a in range(n_actions)}
        self.subset_idx = subset_idx
        self.n_actions = n_actions

    def train(self, X, A, Y):
        for a in range(self.n_actions):
            # select data corresponding to the action a
            X_a, Y_a = X[A == a], Y[A == a]
            self.model[a] = self.model[a].fit(X_a, Y_a)

    def get_actions(self, X):
        pred_actions = np.empty(shape=(X.shape[0], self.n_actions), dtype=np.float32)
        for a in range(self.n_actions):
            pred_actions[:, a] = self.model[a].pred(X)

        return np.argmax(pred_actions, axis=1)
