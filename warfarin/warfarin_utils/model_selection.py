from sklearn.model_selection import KFold
import numpy as np
from joblib import Parallel, delayed
from tqdm.auto import tqdm
import pandas as pd

from warfarin_utils.utils import get_reward


def find_opitmal_set(X, A, Y, init_pi,
                     DMpolicy,
                     candidate_subsets,
                     subsets):
    kf = KFold(n_splits=5)

    def inner_fn(s_name, X, A, Y):
        subset_idx = subsets[s_name]
        exp_rewards = []
        for train_idx, test_idx in kf.split(X):
            X_train, A_train, Y_train, init_pi_train = X[train_idx], A[train_idx], Y[train_idx], init_pi[train_idx]
            X_test, A_test, Y_test = X[test_idx], A[test_idx], Y[test_idx]
            dm_policy = DMpolicy(subset_idx, n_actions=3)
            dm_policy.train(X=X_train, A=A_train, Y=Y_train, init_pi=init_pi_train)
            pred_actions, exp_reward = dm_policy.get_actions(X_test, return_reward=True)
            exp_rewards += [exp_reward]

        return s_name, np.mean(exp_rewards)

    subset_rewards = Parallel(n_jobs=min(20, len(candidate_subsets)))(
        delayed(inner_fn)(s_name, X, A, Y)
        for s_name in tqdm(candidate_subsets.keys(), position=0, leave=True))
    # unzip
    subset_rewards = list(zip(*subset_rewards))

    return pd.Series(index=subset_rewards[0], data=subset_rewards[1])


def get_test_rewards(X, A, Y, E, optimal_dose,
                     init_pi, DMpolicy, candidate_subsets):
    def inner_fn(subset, X_train, A_train, Y_train, init_pi_train, X_test, Y_test):
        subset_name, subset_idx = subset
        rf_policy = DMpolicy(subset_idx, n_actions=3)
        rf_policy.train(X=X_train, A=A_train, Y=Y_train, init_pi=init_pi_train)
        pred_actions = rf_policy.get_actions(X_test)
        reward = get_reward(pred_actions, Y_test).mean()

        return test_E, reward, subset_name

    ret_df = pd.DataFrame(columns=['test_env', 'reward', 'subset'])

    for test_E in pd.unique(E):
        train_idx = (E != test_E)
        X_train, A_train, Y_train, init_pi_train = X[train_idx], A[train_idx], Y[train_idx], init_pi[train_idx]
        X_test, Y_test = X[~train_idx], optimal_dose[~train_idx]

        subset_reward = Parallel(n_jobs=min(20, len(candidate_subsets)))(
            delayed(inner_fn)(subset, X_train, A_train, Y_train, init_pi_train, X_test, Y_test)
            for subset in tqdm(candidate_subsets.items(), position=0, leave=True))

        df_e = pd.DataFrame(data=subset_reward, columns=['test_env', 'reward', 'subset'])
        ret_df = ret_df.append(df_e, ignore_index=True)

    return ret_df


def get_test_rewards_E(X, A, Y, E, optimal_dose,
                       init_pi, DMpolicy, candidate_subsets):
    def inner_fn(subset, X_train, A_train, Y_train, init_pi_train, X_test, Y_test):
        subset_name, subset_idx = subset
        rf_policy = DMpolicy(subset_idx, n_actions=3)
        rf_policy.train(X=X_train, A=A_train, Y=Y_train, init_pi=init_pi_train)
        pred_actions = rf_policy.get_actions(X_test)
        reward = get_reward(pred_actions, Y_test).mean()

        return test_E, reward, subset_name

    ret_df = pd.DataFrame(columns=['test_env', 'reward', 'subset'])

    for test_E in pd.unique(E):
        train_idx = (E != test_E)
        X_train, A_train, Y_train, init_pi_train = X[train_idx], A[train_idx], Y[train_idx], init_pi[train_idx]
        X_test, Y_test = X[~train_idx], optimal_dose[~train_idx]

        subset_reward = Parallel(n_jobs=min(20, len(candidate_subsets)))(
            delayed(inner_fn)(subset, X_train, A_train, Y_train, init_pi_train, X_test, Y_test)
            for subset in tqdm(candidate_subsets[test_E].items(), position=0, leave=True))

        df_e = pd.DataFrame(data=subset_reward, columns=['test_env', 'reward', 'subset'])
        ret_df = ret_df.append(df_e, ignore_index=True)

    return ret_df
