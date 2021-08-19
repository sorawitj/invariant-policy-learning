import itertools
from functools import reduce
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from scipy.special import softmax

def get_action_value_map(optimal_dose):
    low_median = np.median(optimal_dose[optimal_dose <= np.quantile(optimal_dose, 0.33)])
    medium_median = np.median(
        optimal_dose[
            (optimal_dose > np.quantile(optimal_dose, 0.33)) | (optimal_dose <= np.quantile(optimal_dose, 0.66))])
    high_median = np.median(optimal_dose[optimal_dose > np.quantile(optimal_dose, 0.66)])

    mapping = {0: low_median, 1: medium_median, 2: high_median}

    return mapping

optimal_dose = np.load('dataset/therapeut_dose.npy')
action_map = get_action_value_map(optimal_dose)

def unzip(comb):
    ret = list(zip(*comb))
    names = reduce(lambda x1, x2: x1 + '+' + x2, ret[0])
    idx = reduce(lambda x1, x2: np.concatenate([x1, x2]), ret[1])

    return names, idx


def vectorized(prob_matrix, items):
    s = prob_matrix.cumsum(axis=0)
    r = np.random.rand(prob_matrix.shape[1])
    k = (s < r).sum(axis=0)
    return items[k]


def create_subsets(feature_idx, min_size=3):
    subsets = {}
    combinations = range(min_size, len(feature_idx) + 1)
    for c in combinations:
        combi = list(itertools.combinations(feature_idx.items(), c))
        subsets.update(dict([unzip(com) for com in combi]))

    return subsets


def gen_action(optimal_dose):
    mean = optimal_dose.mean()
    std = optimal_dose.std()
    actions = np.random.normal(loc=mean, scale=std, size=optimal_dose.shape[0])
    prop = stats.norm.pdf(actions, loc=mean, scale=std)

    return actions, prop

def gen_action_bmi(context, optimal_dose):
    regrs = sm.OLS(optimal_dose, sm.add_constant(context)).fit()
    pred = regrs.predict(sm.add_constant(context))

    # prop actions
    scores = np.array([1 / np.abs(pred - action_map[0]),
                       1 / np.abs(pred - action_map[1]),
                       1 / np.abs(pred - action_map[2])])
    # temp param
    temp = 0.5
    props = softmax(temp * scores, axis=0)
    actions = vectorized(props, np.arange(3))

    return actions, props[actions, np.arange(len(actions))]


def gen_action_random(optimal_dose):
    action_size = 3
    actions = np.random.choice(action_size, size=optimal_dose.shape[0])
    prop = np.repeat(1 / action_size, repeats=optimal_dose.shape[0])

    return actions, prop


def get_reward(actions, optimal_dose):
    median_doses = np.array([action_map[a] for a in actions])

    return -np.abs(optimal_dose - median_doses)


def get_regret(actions, optimal_dose):
    median_doses = np.array([action_map[a] for a in actions])
    reward = -np.abs(optimal_dose - median_doses)
    optimal_reward = np.array([np.max(-np.abs(y - np.array([action_map[a] for a in range(3)]))) for y in optimal_dose])

    return optimal_reward - reward