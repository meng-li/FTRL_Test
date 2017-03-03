# -*- coding:utf-8 -*-

from math import sqrt, exp

class FTRLProximal(object):

    def __init__(self, alpha, beta, l1, l2, D, interSection):
        # parameters
        self.alpha = alpha
        self.beta = beta
        self.l1 = l1
        self.l2 = l2
        # feature hashing parameters
        self.D = D
        self.interSection = interSection
        # model
        # n: squared sum of past gradients
        # z: weights
        # w: lazy weights
        self.n = [0. ] * D
        self.z = [0. ] * D
        self.w = {}


    def _indices(self, x):
        # bias
        yield 0
        # non-zero features of sample
        for index in x:
            yield index
        # feature intersection in 2-degree
        if self.interSection:
            x = sorted(x)
            for i in xrange(len(x)):
                for j in xrange(i+1, len(x)):
                    yield abs(hash(str(x[i] + '_' + x[j]))) % self.D

    def predict(self, x):
        n = self.n
        z = self.z
        w = {}
        # inner product of w and x
        wTx = 0.
        for i in self._indices(x):
            sign = -1. if z[i] < 0 else 1.
            if sign * z[i] <= self.l1:
                w[i] = 0.
            else:
                w[i] = (sign * self.l1 - z[i]) / \
                        ((self.beta + sqrt(n[i]))/self.alpha + self.l2)
            wTx += w[i]
        self.w = w
        return 1. / (1. + exp(-max(min(wTx, 35.), -35.)))

    def update(self, x, p, y):
        alpha = self.alpha
        n = self.n
        z = self.z
        w = self.w
        # gradient under logloss
        g = p - y
        # update z & n
        for i in self._indices(x):
            # sigma_t = 1 / eta_t - 1 / eta_(t-1)
            #         = (beta + sqrt(n[t])) / alpha - (beta + sqrt(n[t-1])) / alpha
            #         = (sqrt(n[t]) - sqrt(n[t-1])) / alpha
            sigma = (sqrt(n[i] + g * g) - sqrt(n[i])) / alpha
            # z_t = z_(t-1) + g_t + (1/eta_t - 1/eta_(t-1)) * w_t
            z[i] += g + sigma * w[i]
            n[i] += g* g


