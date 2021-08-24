from abc import abstractmethod

from scipy.special import softmax
import numpy as np
import statsmodels.api as sm
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import optim
from torch.distributions import Normal
from tqdm.auto import tqdm


def vectorized(prob_matrix, items):
    s = prob_matrix.cumsum(axis=0)
    r = np.random.rand(prob_matrix.shape[1])
    k = (s < r).sum(axis=0)
    return items[k]


class Policy(object):

    def __init__(self, reg_model, subset, temp, name=None):
        self.name = name
        self.reg_model = reg_model
        self.n_actions = self.reg_model.shape[0]
        self.S = subset
        self.temp = temp

    def get_prob(self, X, A=None):
        p = softmax((1 / self.temp) * self.reg_model.dot(sm.tools.add_constant(X[:, self.S]).T), axis=0)
        if A is None:
            return p
        else:
            return p[A, np.arange(len(A))]

    def get_actions(self, X):
        p = self.get_prob(X)
        A = vectorized(p, np.arange(self.n_actions))
        return A, p[A, np.arange(len(A))]


class RandomPolicy(Policy):

    def __init__(self, n_actions):
        self.n_actions = n_actions
        self.reg_model = np.ones(n_actions)[:, np.newaxis]

        super().__init__(self.reg_model, [], 1.)


class DPolicy(nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, X):
        pass

    def get_prob(self, X, A=None):
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X.astype(np.float32))
        p = self.forward(X)
        if A is None:
            return p
        else:
            return p[np.arange(len(A)), A]

    def get_actions(self, X):
        p = self.get_prob(X).detach().numpy()
        A = vectorized(p.T, np.arange(self.n_actions))
        return A, p[np.arange(len(A)), A]


class NaivePolicy(DPolicy):
    def __init__(self, n_actions, name='Naive'):
        self.name = name
        self.n_actions = n_actions
        super(NaivePolicy, self).__init__()
        self.score = nn.Parameter(torch.ones(n_actions, dtype=torch.float32), requires_grad=True)

    def forward(self, X):
        return F.softmax(self.score.repeat(X.shape[0], 1), dim=1)

class LinearPolicy(DPolicy):
    def __init__(self, target_set, n_actions, name='Linear'):
        self.name = name
        self.n_actions = n_actions
        self.target_set = target_set
        super(LinearPolicy, self).__init__()
        self.linear = nn.Linear(len(target_set), n_actions)

    def forward(self, X):
        logits = self.linear(X[:, self.target_set])
        return F.softmax(logits, dim=1)


class ContinousPolicy(nn.Module):

    def __init__(self, target_set, name='LinearCon'):
        super(ContinousPolicy, self).__init__()
        self.name = name
        self.target_set = target_set
        self.linear = nn.Linear(len(target_set), 1)

    def forward(self, X):
        mean = self.linear(X[:, self.target_set]).flatten()

        return mean


class Gaussian_Policy(nn.Module):
    '''
    Gaussian policy that consists of a neural network with 1 hidden layer that
    outputs mean and log std dev (the params) of a gaussian policy
    '''

    def __init__(self, target_set, hidden_size):
        super(Gaussian_Policy, self).__init__()

        self.target_set = target_set

        self.linear = nn.Linear(len(target_set), hidden_size)
        self.mean = nn.Linear(hidden_size, 1)
        self.log_std = nn.Linear(hidden_size, 1)

    def forward(self, X):
        # forward pass of NN
        hidden = F.relu(self.linear(X[:, self.target_set]))

        mean = self.mean(hidden).flatten()
        log_std = self.log_std(hidden)
        log_std = torch.clamp(log_std, min=-2, max=5)  # We limit the variance by forcing within a range of -2,20
        std = log_std.exp().flatten()

        return mean, std


to_torch = lambda x: torch.from_numpy(x.astype(np.float32))


class PolicyGrad(object):

    def __init__(self, target_set, hidden_size=10, lr=1e-4, std=1.0):
        self.target_set = target_set
        self.hidden_size = hidden_size
        self.lr = lr
        self.policy = ContinousPolicy(target_set)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.std = std

    def reset(self):
        self.policy = ContinousPolicy(self.target_set)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)

    def train(self, X, A, R, pi_logging, n_iter=2000, print_every_i=50):
        if isinstance(X, np.ndarray):
            X = to_torch(X)
        if isinstance(A, np.ndarray):
            A = to_torch(A)
        if isinstance(R, np.ndarray):
            R = to_torch(R)
        if isinstance(pi_logging, np.ndarray):
            pi_logging = to_torch(pi_logging)

        # store mean and std
        self.mean_A = A.mean()
        self.std_A = A.std()
        # standardize actions
        A = (A - self.mean_A) / self.std_A

        running_return = None
        prev_running_return = -np.inf
        stop_count = 0
        pbar = tqdm(range(n_iter), position=0, leave=True)
        for i in pbar:
            weighted_rewards, log_prob = self.pred_rewards(X, A, R, pi_logging, train=True)
            policy_loss = -log_prob * weighted_rewards
            self.optimizer.zero_grad()
            policy_loss = policy_loss.sum()
            policy_loss.backward()
            self.optimizer.step()

            exp_rewards = torch.mean(weighted_rewards)

            if running_return is None:
                running_return = exp_rewards
            else:
                running_return = 0.05 * exp_rewards + (1 - 0.05) * running_return

            if i % print_every_i == 0:
                pbar.set_description(
                    'Iteration {}\t Exp Return: {}\t Running Return: {}'.format(i, exp_rewards.detach().numpy(),
                                                                                running_return))

                if exp_rewards <= prev_running_return:
                    stop_count += 1
                else:
                    prev_running_return = running_return

                if stop_count > 10:
                    print("Stop! exp returns does not improve")
                    return exp_rewards
        return exp_rewards

    def pred_rewards(self, X, A, R, pi_logging, train=True):
        if isinstance(X, np.ndarray):
            X = to_torch(X)
        if isinstance(A, np.ndarray):
            A = to_torch(A)
        if isinstance(R, np.ndarray):
            R = to_torch(R)
        if isinstance(pi_logging, np.ndarray):
            pi_logging = to_torch(pi_logging)

        mean = self.policy(X)
        mean = (mean - mean.mean()) / mean.std()
        # create normal distribution
        normal = Normal(mean, self.std)
        log_prob = normal.log_prob(A)
        pi_target = torch.exp(log_prob)
        pi_target = pi_target / pi_target.sum()
        w = pi_target / pi_logging
        if train:
            returns = (R - R.mean())
            return returns * w.detach(), log_prob
        else:
            return R * w.detach()

    def get_actions(self, X):
        if isinstance(X, np.ndarray):
            X = to_torch(X)

        pred = self.policy(X)
        pred = (pred - pred.mean()) / pred.std()
        pred = pred * self.std_A + self.mean_A

        return pred
