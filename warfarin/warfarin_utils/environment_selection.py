import itertools

import pandas as pd
import numpy as np
from scipy import stats


def get_dist(race_info, race_df):
    freq_df = pd.Series(race_info).value_counts().reset_index(name='freq').rename(columns={'index': 'Race'})
    freq_df = race_df.merge(freq_df, how='left', on='Race')
    freq_df['dist'] = freq_df['freq'] / freq_df['freq'].sum()
    freq_df['dist'] = freq_df['dist'].fillna(0)
    return freq_df['dist'].to_numpy()


def find_dis_matrix_race(envs, race_info, race_df):
    env_idx = {e: i for i, e in enumerate(pd.unique(envs))}
    race_idx = [i for i in range(len(race_df))]

    dissim_matrix = np.empty(shape=(len(env_idx), len(env_idx)))

    for e1, e2 in itertools.combinations(pd.unique(envs), 2):
        # create and array with cardinality 3 (your metric space is 3-dimensional and
        # where distance between each pair of adjacent elements is 1
        e1_dist = get_dist(race_info[envs == e1], race_df)
        e2_dist = get_dist(race_info[envs == e2], race_df)

        wd = stats.wasserstein_distance(race_idx, race_idx, e1_dist, e2_dist)

        dissim_matrix[env_idx[e1], env_idx[e2]] = wd
        dissim_matrix[env_idx[e2], env_idx[e1]] = wd

    for e in pd.unique(envs):
        e_dist = get_dist(race_info[envs == e], race_df)

        wd = stats.wasserstein_distance(race_idx, race_idx, e_dist, e_dist)

        dissim_matrix[env_idx[e], env_idx[e]] = wd

    return dissim_matrix, env_idx