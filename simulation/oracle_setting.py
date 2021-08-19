# Add invariant test into the path
import sys, os

# add invariant test module
sys.path.append(os.path.abspath(os.path.join('..', 'invariant_test_utils')))

from tqdm import tqdm
from sim_utils.environment import Environment
from invariant_test import SIR
from sim_utils.learner import IPWLearner, train_lsq
from sim_utils.policy import RandomPolicy, Policy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from multiprocessing import Pool, cpu_count

n_actions = 3
n_iters = 400
s_size = 10000
n_sampling = 1000
batch_size = 100
true_coef = False
sig_level = 0.05
inv_seed = 111
seed = 0
known_set = True
target_sets = {'X2': [1],
               'X1,X2': [0, 1]}
np.random.seed(seed)


# TODO change s_size such that it's the total number of observations (not per environment)

def check_invariant(tester, target_set, target_policy, m,
                    X, A, R, P, E, alpha=.05):
    target_P = target_policy.get_prob(X, A)

    w = target_P / P

    num_env = E.max() + 1
    s_idx = []
    for e in range(num_env):
        idx = np.where(E == e)[0]
        e_idx = SIR(idx, w[idx], int(m / num_env), True)
        s_idx += [e_idx]
    s_idx = np.concatenate(s_idx)

    _, pvalue = tester.get_test_stat(X, A, R, E,
                                     target_set=target_set,
                                     s_idx=s_idx,
                                     true_coef=true_coef)
    return pvalue >= alpha


def arr_toDF(X, A, R, w=None, eval=False):
    dat = {}
    for i in range(X.shape[1]):
        dat['X{}'.format(i + 1)] = X[:, i]
    if eval:
        dat['R'] = R.max(1)
        for i in range(n_actions):
            dat['A{}'.format(i)] = i
    else:
        dat['R'] = R[np.arange(R.shape[0]), A]
        dat['A'] = A
    if w is not None:
        dat['w'] = w

    return pd.DataFrame(data=dat)


n_envs = [2, 6]
learner = {n_env: {} for n_env in n_envs}
corr_train = {}
for n_env in n_envs:
    random_policy = RandomPolicy(n_actions)
    loggin_env = Environment(1, n_actions, 1, 1)
    X_log, A_log, R_log, _, _ = loggin_env.gen_data(random_policy, 100000)
    loggin_policy = Policy(train_lsq(X_log, R_log, target_sets['X1,X2']), target_sets['X1,X2'], 2.)

    train_env = Environment(n_env, n_actions, inv_seed=inv_seed, non_inv_seed=seed, train=True)
    X, A, R, P, E = train_env.gen_data(loggin_policy, s_size)
    # get correlation between U and X1 in the training environments
    corr_train[n_env] = train_env.get_corr()
    # tester = PoolRegress(n_env, n_actions, train_env)
    # trainer = PowerOpTrainer(n_iters, batch_size, n_sampling, tester, true_coef)

    w = 1. / P
    train_df = arr_toDF(X, A, R, w, eval=False)
    for target_name, target_set in target_sets.items():
        ipwLearner = IPWLearner(target_set, n_actions)
        ipwLearner.train(train_df)
        learner[n_env][target_name] = ipwLearner


# function for computing experiment
def compute_experiment(i):
    ret_dict = {'Distance': [], 'Policy': [], 'Regret': [], 'n_env': []}
    for n_env in n_envs:
        test_env = Environment(1, n_actions, inv_seed=inv_seed, non_inv_seed=(i, n_env), train=False)
        # get correlation between U and X1 in the test environment
        # corr_test = test_env.get_corr()
        # e_diff = np.abs(corr_train[n_env] - corr_test).mean()
        e_diff = np.linalg.norm(train_env.e - test_env.e)
        # random_policy = RandomPolicy(n_actions)
        np.random.seed((i, n_env))
        X_test, A_test, R_test, _, _ = test_env.gen_data(random_policy, 10000)
        test_df = arr_toDF(X_test, A_test, R_test, eval=True)

        for target_name, target_set in target_sets.items():
            ret_dict['n_env'] += [n_env]
            ret_dict['Distance'] += [e_diff]
            ret_dict['Policy'] += [target_name]
            ret_dict['Regret'] += [learner[n_env][target_name].eval(test_df, R_test)]

    return ret_dict


## Conduct multiple experiments with multiprocessing
if __name__ == '__main__':
    repeats = 1000

    # Multiprocess
    pool = Pool(cpu_count() - 2)
    res = np.array(
        list(tqdm(pool.imap_unordered(compute_experiment, range(repeats)), total=repeats)))
    pool.close()

    temp_dict = {k: [val for dic in res for val in dic[k]] for k in res[0].keys()}
    df_regrets = pd.DataFrame(temp_dict)

    sns.set(style='white', font_scale=1.2)
    sns.set_palette("tab10")

    g = sns.relplot(
        data=df_regrets, x="Distance", y="Regret",
        col="n_env", hue="Policy", col_wrap=2,
        kind="scatter", s=60, alpha=.4, aspect=1.,
        height=3, legend='full'
    )
    leg = g._legend
    leg.set_bbox_to_anchor([1.01, 0.8])
    g.set_titles('#training envs: {col_name}')

    plt.tight_layout()

    plt.savefig('results/oracle_setting.pdf')
