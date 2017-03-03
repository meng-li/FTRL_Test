# -*- coding:utf-8 -*-

from math import exp

SIGMOID_BOUND = 35.

class LogisticRegression(object):

    def __init__(self, alpha, D, interSection=False):
        self.alpha = alpha
        self.D = D
        self.w = [0. ] * D
        self.interSection = interSection

    def _indices(self, x):
        for idx in x:
            yield idx

        if self.interSection:
            for i in xrange(1, len(x)):
                for j in xrange(i+1, len(x)):
                    yield (i * j) % D

    def predict(self, x):
        alpha = self.alpha
        w = self.w

        # inner product of w and x
        wTx = sum(w[i] * 1. for i in self._indices(x))
        score = 1. / (1. + exp(
            -max(min(wTx, SIGMOID_BOUND),
            -1. * SIGMOID_BOUND)))
        return score

    def update(self, x, p, y):
        alpha = self.alpha
        w = self.w
        g = p - y
        for i in self._indices(x):
            w[i] += g * alpha
