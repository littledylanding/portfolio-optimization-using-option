import numpy as np
import cvxpy as cp


class mvp:
    def __init__(self, price, var=None, rolling_period=250, type='normal', rebalance=1):
        self.price = price
        self.ret = []
        self.day = rolling_period
        self.rolling_period = rolling_period
        self.type = type
        self.weights = None
        self.rebalance = rebalance
        self.turnover = 0
        self.commission = 0.0004
        self.value = 10 ** 5
        self.shares = None
        self.var = var

    def before(self):
        data = self.price.iloc[self.day - self.rolling_period:self.day, :]
        if self.type == 'equal':
            w = np.ones(len(self.price.columns)) / len(self.price.columns)
        else:
            if self.type == 'normal':
                cov = data.cov().values
            else:
                corr = data.corr().values
                var = np.zeros((len(self.price.columns), len(self.price.columns)))
                diag = self.var.iloc[self.day-1].values
                ind = np.diag_indices_from(var)
                var[ind] = diag
                cov = var.dot(corr.dot(var))
            if self.type != 'commission':
                w = cp.Variable(len(self.price.columns))
                objective = cp.Minimize(cp.quad_form(w, cov))
                constraints = [0 <= w, w <= 1, cp.sum(w) == 1]
                prob = cp.Problem(objective, constraints)
                prob.solve()
            w = w.value
        self.weights = w
        if self.day != self.rolling_period:
            data = self.price.iloc[self.day - 1, :].values
            new_w = self.shares * data / np.sum(self.shares * data)
            dw = w - new_w
            self.turnover += np.sum(np.abs(dw)) / (
                    len(self.price) - self.rebalance - self.rolling_period)
            c = np.sum(np.abs(dw)) * self.commission
            self.ret[-1] -= c
            self.value *= 1 + self.ret[-1]
        self.shares = self.value * self.weights / self.price.iloc[self.day - 1, :].values

    def after(self):
        new_value = np.sum(self.shares * self.price.iloc[self.day, :].values)
        r = (new_value - self.value) / self.value
        self.ret.append(r)
        self.day += self.rebalance

    def trade(self):
        for i in range(self.rolling_period, len(self.price)):
            self.before()
            self.after()
