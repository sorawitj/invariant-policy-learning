import torch
import torch.optim as optim
import numpy as np
import sys, os

# add invariant test module
sys.path.append(os.path.abspath(os.path.join('..', 'invariant_test_utils')))

from invariant_test import SIR


class PowerOpTrainer(object):

    def __init__(self, n_iters, batch_size,
                 n_sampling, tester, true_coef=False):
        self.n_iters = n_iters
        self.batch_size = batch_size
        self.n_sampling = n_sampling
        self.tester = tester
        self.true_coef = true_coef

    def train(self, X, A, R, E, P,
              policy, target_set):
        optimizer = optim.Adam(policy.parameters(), lr=1e-1)
        prev_running_pval = 1.
        running_pval = None
        eps = np.finfo(np.float32).eps.item()

        for i in range(self.n_iters):
            policy_loss = []
            pvals = []
            log_probs = []
            for _ in range(self.batch_size):
                target_P = policy.get_prob(X, A)
                w = target_P / torch.from_numpy(P)
                w = w / torch.sum(w)

                s_idx = SIR(w.detach().numpy(), self.n_sampling * self.tester.n_env, True)
                s_prob = w[s_idx]

                t_stat, pvalue = self.tester.get_test_stat(X, A, R, E,
                                                           target_set=target_set,
                                                           s_idx=s_idx,
                                                           true_coef=self.true_coef)
                pvals.append(pvalue)
                log_probs.append(torch.log(s_prob).sum())

            pvals = torch.tensor(pvals)
            avg_pval = pvals.mean()
            if running_pval is None:
                running_pval = avg_pval
            else:
                running_pval = 0.05 * avg_pval + (1 - 0.05) * running_pval
            pvals = (pvals - avg_pval) / (pvals.std() + eps)
            for log_prob, adv in zip(log_probs, pvals):
                policy_loss.append(log_prob * adv)
            optimizer.zero_grad()
            policy_loss = torch.stack(policy_loss).mean()
            policy_loss.backward()
            optimizer.step()

            if i % 40 == 0:
                print('Iteration {}\t Avg p-values: {}\t Running p-values: {}'.format(i, avg_pval, running_pval))
                if i > 60:
                    if running_pval >= prev_running_pval:
                        print("Stop! running p-value does not decrease")
                        return prev_running_pval
                prev_running_pval = running_pval

        return running_pval
