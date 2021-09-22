# Add invariant test into the path
import sys, os

# add invariant test module
sys.path.append(os.path.abspath(os.path.join('..', 'invariant_test_utils')))

from sklearn.linear_model import LinearRegression
from tqdm import tqdm
from sim_utils.environment import Environment
from sim_utils.learner import train_lsq
from sim_utils.policy import RandomPolicy, Policy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from multiprocessing import Pool, cpu_count
from invariant_test import invariance_test_actions_resample, fit_m, rate_fn

n_actions = 3
inv_seed = 111
seed = 0
target_sets = {r'$\emptyset$': [],
               'X1': [0],
               'X2': [1],
               'X1,X2': [0, 1]}
np.random.seed(seed)

n_envs = [2, 6]
s_sizes = [1000, 3000, 9000, 27000, 81000]

random_policy = RandomPolicy(n_actions)
loggin_env = Environment(1, n_actions, 1, 1)
X_log, A_log, R_log, _, _ = loggin_env.gen_data(random_policy, int(1e4))
loggin_policy = Policy(train_lsq(X_log, R_log, target_sets['X1,X2']), target_sets['X1,X2'], 1.75)


def verify_policies(s_size,
                    n_env,
                    draw_seed,
                    loggin_policy=loggin_policy,
                    inv_seed=inv_seed,
                    non_inv_seed=seed,
                    alpha=0.05):
    model = LinearRegression()
    train_size = int(s_size / n_env)

    train_env = Environment(n_env, n_actions, inv_seed=inv_seed, non_inv_seed=non_inv_seed, train=True)
    np.random.seed(draw_seed)
    X, A, R, P, E = train_env.gen_data(loggin_policy, train_size)
    # get target and weights
    Y = R[np.arange(len(R)), A]
    W = 1 / P

    is_invs = {}
    m = fit_m(A, X, W, n_iter=10)
    rate = rate_fn(1, m / s_size)
    for subset in target_sets.items():
        target_name, p_val = invariance_test_actions_resample(model, subset,
                                                              X, A, Y, E, W, rate)
        is_invs[target_name] = p_val >= alpha

    return is_invs


# function for computing experiment
def compute_experiment(i):
    ret_dict = {'Acceptance Rate': [], 'Policy': [], 'n_env': [], 'Sample Size': []}
    for s_size in s_sizes:
        for n_env in n_envs:
            is_invs = verify_policies(s_size, n_env, draw_seed=(i, s_size, n_env))

            for policy_name, is_inv in is_invs.items():
                ret_dict['Acceptance Rate'] += [is_inv]
                ret_dict['n_env'] += [n_env]
                ret_dict['Policy'] += [policy_name]
                ret_dict['Sample Size'] += [s_size]

    return ret_dict


## Conduct multiple experiments with multiprocessing
if __name__ == '__main__':
    repeats = 500

    # Multiprocess
    pool = Pool(cpu_count() - 2)
    res = np.array(
        list(tqdm(pool.imap_unordered(compute_experiment, range(repeats)), total=repeats)))
    pool.close()

    temp_dict = {k: [val for dic in res for val in dic[k]] for k in res[0].keys()}
    df_regrets = pd.DataFrame(temp_dict)

    sns.set(font_scale=1.2, style='white', palette=sns.set_palette("tab10"))

    g = sns.relplot(
        data=df_regrets, x="Sample Size", y="Acceptance Rate",
        col="n_env", hue="Policy", kind="line", marker='o', markersize=6,
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
    plt.tight_layout()

    plt.savefig('results/invariant_test.pdf')
