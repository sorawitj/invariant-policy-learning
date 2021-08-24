import torch
import torch.optim as optim
import numpy as np
import sys, os

# add invariant test module
sys.path.append(os.path.abspath(os.path.join('..', 'invariant_test_utils')))

from invariant_test import resampling, invariance_test


class PowerOpTrainer(object):

    def __init__(self, n_iters, batch_size,
                 rate, model, true_coef=False):
        self.n_iters = n_iters
        self.batch_size = batch_size
        self.rate = rate
        self.test_model = model
        self.true_coef = true_coef

    def train(self, X, A, Y, E, P,
              policy, subset):
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
                W = target_P / torch.from_numpy(P)
                W = W / torch.sum(W)

                s_idx = resampling(self.rate, E, W=W.detach().numpy(), replace=False)
                s_prob = W[s_idx]

                X_s, Y_s, E_s = X[s_idx], Y[s_idx], E[s_idx]

                _, pvalue = invariance_test(self.test_model, subset, X_s, Y_s, E_s)

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
