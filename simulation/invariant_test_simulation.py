# Add invariant test into the path
import pickle
import sys, os

# add invariant test module
from sim_utils.power_op_trainer import PowerOpTrainer

sys.path.append(os.path.abspath(os.path.join('..', 'invariant_test_utils')))

from sklearn.linear_model import LinearRegression
from tqdm import tqdm
from sim_utils.environment import Environment
from sim_utils.learner import train_lsq
from sim_utils.policy import RandomPolicy, Policy, LinearPolicy, NaivePolicy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from multiprocessing import Pool, cpu_count
from invariant_test import invariance_test_actions_resample, fit_m, rate_fn, invariance_test_resample

# main config
n_actions = 3
inv_seed = 111
seed = 0
target_sets = {r'$\emptyset$': [],
               'X1': [0],
               'X2': [1],
               'X1,X2': [0, 1]}
np.random.seed(seed)

s_sizes = [300, 900, 2700, 8100]
methods = ['unif', 'opt']


def verify_policies(s_size,
                    method,
                    draw_seed,
                    loggin_policy,
                    inv_seed=inv_seed,
                    non_inv_seed=seed,
                    alpha=0.05):
    if method == 'opt':
        new_s_size = s_size * 0.7
    elif method == 'unif':
        new_s_size = s_size

    train_size = int(new_s_size / n_env)
    train_env = Environment(n_env, n_actions, inv_seed=inv_seed, non_inv_seed=non_inv_seed, train=True)
    np.random.seed(draw_seed)
    X, A, R, P, E = train_env.gen_data(loggin_policy, train_size)
    # get target and weights
    Y = R[np.arange(len(R)), A]

    is_invs = {}

    for subset in target_sets.items():
        subset_name, subset_idx = subset
        if method == 'opt':
            new_P = opt_policy[subset_name][s_size].get_prob(X, A)
            W = new_P.detach().numpy() / P
        elif method == 'unif':
            W = 1 / P

        if subset_idx != [0, 1]:
            if method == 'opt':
                context_idx = [s for s in [0, 1] if s not in subset_idx]
            elif method == 'unif':
                context_idx = [0, 1]
            m = fit_m(A, X[:, context_idx], W, n_iter=10)
            rate = rate_fn(1, m / new_s_size)
        else:
            rate = rate_fn(1, 1)
        if method == 'opt':
            target_name, p_val = invariance_test_resample(test_model, subset, X, A, Y, E, W, rate)
        elif method == 'unif':
            target_name, p_val = invariance_test_actions_resample(test_model, subset,
                                                                  X, A, Y, E, W, rate)
        is_invs[target_name] = p_val >= alpha

    return is_invs


# function for computing experiment
def compute_experiment(i, loggin_policy):
    ret_dict = {'Acceptance Rate': [], 'Policy': [], 'Method': [], 'Sample Size': []}
    for s_size in s_sizes:
        for method in methods:
            is_invs = verify_policies(s_size, method, loggin_policy=loggin_policy, draw_seed=(i, s_size, n_env))

            for policy_name, is_inv in is_invs.items():
                ret_dict['Acceptance Rate'] += [is_inv]
                ret_dict['Method'] += [method]
                ret_dict['Policy'] += [policy_name]
                ret_dict['Sample Size'] += [s_size]

    return ret_dict


## Conduct multiple experiments with multiprocessing
if __name__ == '__main__':

    random_policy = RandomPolicy(n_actions)
    loggin_env = Environment(1, n_actions, 1, 1)
    X_log, A_log, R_log, _, _ = loggin_env.gen_data(random_policy, int(1e4))
    loggin_policy = Policy(train_lsq(X_log, R_log, target_sets['X1,X2']), target_sets['X1,X2'], 3)
    test_model = LinearRegression()
    n_env = 6
    opt_policy = {k: {} for k in target_sets.keys()}
    load_opt_dict = True

    if load_opt_dict:
        a_file = open("opt_policy.pkl", "rb")
        opt_policy = pickle.load(a_file)
    else:
        for s_size in s_sizes:
            rate = rate_fn(.7, 1)
            train_env = Environment(n_env, n_actions, inv_seed=inv_seed, non_inv_seed=seed, train=True)
            trainer = PowerOpTrainer(200, 128, rate, test_model)
            for subset in target_sets.items():
                train_size = int(s_size * 0.3 / n_env)
                subset_name, subset_idx = subset
                if len(subset_idx) > 0:
                    test_policy = LinearPolicy(subset_idx, n_actions)
                else:
                    test_policy = NaivePolicy(n_actions)
                X, A, R, P, E = train_env.gen_data(loggin_policy, train_size)
                Y = R[np.arange(len(R)), A]
                trainer.train(X, A, Y, E, P, test_policy, subset)
                opt_policy[subset_name][s_size] = test_policy

    repeats = 500

    # Multiprocess
    pool = Pool(cpu_count() - 2)
    res = np.array(
        list(tqdm(pool.imap_unordered(lambda x: compute_experiment(x, loggin_policy), range(repeats))
                  , total=repeats)))
    pool.close()

    temp_dict = {k: [val for dic in res for val in dic[k]] for k in res[0].keys()}
    df_regrets = pd.DataFrame(temp_dict)

    sns.set(font_scale=1.2, style='white', palette=sns.set_palette("tab10"))

    g = sns.relplot(
        data=df_regrets, x="Sample Size", y="Acceptance Rate",
        col="Method", hue="Policy", kind="line", marker='o', markersize=6,
        height=3.25, aspect=1, alpha=.6
    )
    g.set(xscale="log")
    g.set_titles('#training envs: {col_name}')
    g.set_xlabels("Sample Size (Total)")

    leg = g._legend
    leg.set_bbox_to_anchor([1., 0.55])
    for ax in g.axes[0]:
        ax.axhline(0.95, ls='--', color='black', label='95% level', linewidth=0.85, alpha=0.7)
        plt.legend(bbox_to_anchor=(1.5, 1.2))

    # plt.savefig('results/invariant_test_utils.pdf')
