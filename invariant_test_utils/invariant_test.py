import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import kruskal, mannwhitneyu

def SIR(idx, w, m, replace):
    sample_idx = np.random.choice(idx, size=m, replace=replace, p=w / np.sum(w))
    return sample_idx


def resampling(rate, E, W, replace=False):
    s_idx = []
    for e in pd.unique(E):
        idx = np.where(E == e)[0]
        m = rate(len(idx))
        e_idx = SIR(idx, W[idx], int(m), replace)
        s_idx += [e_idx]
    s_idx = np.concatenate(s_idx)
    return s_idx


# Define rates for resampling
def rate_fn(pow, c=1):
    def f(n): return c * n ** pow

    return f

def fit_m(A, context, W, n_iter=20):
    def inner_pval(context, s_idx):
        pval = []
        for j in range(context.shape[1]):
            pval += [kruskal(*[context[s_idx, j][A[s_idx] == a] for a in pd.unique(A)]).pvalue]

        return context.shape[1] * np.min(pval)

    threshold = np.quantile(np.random.uniform(size=(100000, n_iter)).mean(-1), 0.001)

    n = A.shape[0]
    m_size = np.linspace(n ** 0.6, n ** 1.0, 20)

    ret_m = m_size[0]
    for m in m_size:
        re_idx = [SIR(len(A), W, int(m), False) for _ in range(n_iter)]

        pvals = [inner_pval(context, s_idx) for s_idx in re_idx]

        if np.mean(pvals) > threshold:
            ret_m = m
        else:
            break

    return ret_m


def invariance_test(model, subset, X, Y, E):
    subset_name, subset_idx = subset

    # fit the model with the given subset
    # handle empty set
    if len(subset_idx) > 0:
        model.fit_intercept = True
        feature_subset = X[:, subset_idx]
    else:
        model.fit_intercept = False
        feature_subset = np.ones(shape=(X.shape[0], 1))

    model.fit(feature_subset, Y)

    # test for invariance
    residuals = []
    for e in pd.unique(E):
        env_idx = E == e
        pred_e = model.predict(feature_subset[env_idx])
        reward_e = Y[env_idx]
        residuals += [reward_e - pred_e]

    if len(residuals) == 2:
        pval = mannwhitneyu(residuals[0], residuals[1]).pvalue
    else:
        pval = kruskal(*residuals).pvalue

    return subset_name, pval


def invariance_test_actions(model, subset, X, A, Y, E):
    subset_name, subset_idx = subset

    pvals = []
    for a in pd.unique(A):
        # select data corresponding to the action a
        X_a, Y_a, E_a = X[A == a], Y[A == a], E[A == a]
        # skip when Ea has only one group
        if len(pd.unique(E_a)) > 1:
            _, pval = invariance_test(model, subset, X_a, Y_a, E_a)
            pvals += [pval]
        else:
            print('skip')

    final_pval = len(pd.unique(A)) * np.min(pvals)

    return subset_name, final_pval


def invariance_test_actions_resample(model, subset, X, A, Y, E, W,
                                     rate, n_resample_iter=1):
    subset_name, subset_idx = subset

    pvals = []

    for i in range(n_resample_iter):
        # resampling step
        s_idx = resampling(rate, E, W=W, replace=False)
        # resample
        X_s, A_s, Y_s, E_s = X[s_idx], A[s_idx], Y[s_idx], E[s_idx]

        _, pval_i = invariance_test_actions(model, subset, X_s, A_s, Y_s, E_s)

        pvals += [pval_i]

    final_pval = n_resample_iter * np.min(pvals)

    return subset_name, final_pval


def plot_residuals(X, Y, E,
                   model,
                   subset_idx):
    res_e = []
    envs = []
    actions = []
    # select data corresponding to the action a
    # fit the model with the given subset
    feature_subset = X[:, subset_idx]
    model.fit(feature_subset, Y)

    # predictions
    pred = model.predict(feature_subset)
    res = Y - pred.flatten()

    for e in pd.unique(E):
        res_e += [res[E == e]]
        envs += [E[E == e]]

    envs = np.concatenate(envs)
    res = np.concatenate(res_e)

    df = pd.DataFrame({'env': envs, 'pval': res})
    dd = pd.melt(df, id_vars=['env'], value_vars=['pval'], var_name='subset')
    grid = sns.catplot(x='env', y='value', data=dd, hue='subset',
                       kind="box", height=5, aspect=1,
                       whis=2.0, fliersize=4,
                       showmeans=True,
                       meanprops={"marker": "s", "markerfacecolor": "white", "markeredgecolor": "blue", "markersize": 8}
                       )

    grid.axes[0][0].set_ylabel('Residuals', fontsize=15)
    for ax in grid.axes[0]:
        ax.set_ylim([-45, 40])
        ax.set_xlabel('Environment', fontsize=15)


def plot_residuals_fitted(X, Y, E,
                          model,
                          subset_idx):
    # select data corresponding to the action a
    # fit the model with the given subset
    feature_subset = X[:, subset_idx]
    model.fit(feature_subset, Y)

    # predictions
    pred = model.predict(feature_subset)
    res = Y - pred.flatten()

    df = pd.DataFrame({'env': E, 'pred': pred, 'res': res})
    grid = sns.scatterplot(x='pred', y='res', data=df, hue='env', palette=sns.color_palette())

    grid.axes[0][0].set_ylabel('Residuals', fontsize=15)
    for ax in grid.axes[0]:
        ax.set_ylim([-45, 40])
        ax.set_xlabel('Environment', fontsize=15)


def plot_residuals_action(X, A, Y, E, W,
                          model,
                          subset_min,
                          subset_max,
                          rs_rate=None):
    res = []

    if W is not None:
        # resampling step
        s_idx = resampling(rs_rate, E, W=W, replace=False)
        # resample
        X_s, A_s, Y_s, E_s = X[s_idx], A[s_idx], Y[s_idx], E[s_idx]
    else:
        # no resampling
        X_s, A_s, Y_s, E_s = X, A, Y, E

    for subset_idx in [subset_min, subset_max]:

        res_e = []
        envs = []
        actions = []
        for a in pd.unique(A):
            # select data corresponding to the action a
            X_a, A_a, Y_a, E_a = X_s[A_s == a], A_s[A_s == a], Y_s[A_s == a], E_s[A_s == a]
            # fit the model with the given subset
            feature_subset_a = X_a[:, subset_idx]
            model.fit(feature_subset_a, Y_a)

            # predictions
            pred = model.predict(feature_subset_a)
            res_a = Y_a - pred.flatten()

            for e in pd.unique(E):
                res_e += [res_a[E_a == e]]
                envs += [E_a[E_a == e]]
                actions += [A_a[E_a == e]]

        envs = np.concatenate(envs)
        actions = np.concatenate(actions)
        res += [np.concatenate(res_e)]

    df = pd.DataFrame({'env': envs, 'actions': actions, 'min_pval': res[0], 'max_pval': res[1]})
    dd = pd.melt(df, id_vars=['env', 'actions'], value_vars=['min_pval', 'max_pval'], var_name='subset')
    grid = sns.catplot(x='env', y='value', data=dd, hue='subset',
                       col='actions', kind="box", height=5, aspect=1,
                       whis=2.0, fliersize=4,
                       showmeans=True,
                       meanprops={"marker": "s", "markerfacecolor": "white", "markeredgecolor": "blue", "markersize": 8}
                       )

    grid.axes[0][0].set_ylabel('Residuals', fontsize=15)
    for ax in grid.axes[0]:
        ax.set_ylim([-45, 40])
        ax.set_xlabel('Environment', fontsize=15)
