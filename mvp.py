import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp


class mvp:
    def __init__(self, price, rolling_period=250, type='Normal', rebalance=1):
        self.price = price
        self.ret = []
        self.day = rolling_period
        self.rolling_period = rolling_period
        self.type = type
        self.weights = None
        self.rebalance = rebalance
        self.turnover = 0

    def before(self, cov=None):
        data = self.price.iloc[self.day - self.rolling_period:self.day, :]
        mean = data.mean.values
        if self.type != 'Normal':
            cov = data.cov.values
        w = cp.Variable(len(self.price.columns), 1)
        objective = cp.Minimize(w.T @ cov @ w)
        constraints = [0 <= w, w <= 1]
        prob = cp.Problem(objective, constraints)
        prob.solve()
        if self.day != self.rolling_period:
            self.turnover += np.sum(np.abs(w.value - self.weights)) / (
                        len(self.price) - self.rebalance - self.rolling_period)
        self.weights = w.value

    def after(self):
        data = self.price.iloc[self.day - self.rebalance:self.day + 1, :]
        r = data.diff().values
        r = r.dot(self.weights)
        self.ret.append(r)
        self.day += self.rebalance

    def start(self, cov):
        self.before(cov)
        self.after
