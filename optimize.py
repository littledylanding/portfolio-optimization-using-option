import cvxpy as cp
import numpy as np
import pandas as pd

# Read stock prices from Excel file
data = pd.read_excel('stocks.xlsx')
prices = data.iloc[:, 1:]

# Calculate returns
returns = prices.pct_change().dropna()

# Constants
N = len(returns.columns)  # Number of assets
mu = returns.mean().values  # Expected returns (N-vector)
Sigma = returns.cov().values  # Covariance matrix (NxN)
w_bar = np.ones(N) / N  # Example: Existing portfolio (equal weight)
E_0 = 0.01  # Minimum acceptable rate of return
c_B = np.ones(N) * 0.005  # Example: Proportional transaction costs for buying (N-vector)
c_S = np.ones(N) * 0.005  # Example: Proportional transaction costs for selling (N-vector)
U_i = np.ones(N) * 0.1  # Example: Holding constraint upper bounds (N-vector)
U_TRN = 0.15  # Example: Turnover constraint upper bound

# Variables
w = cp.Variable(N)
x_plus = cp.Variable(N)
x_minus = cp.Variable(N)
phi = cp.Variable()

# Objective function
objective = cp.Minimize(cp.quad_form(w, Sigma) / (1 - phi))

# Constraints
constraints = [
    cp.sum(w) + cp.sum(c_B * x_plus) + cp.sum(c_S * x_minus) == 1,
    w.T @ mu >= E_0,
    w - x_plus + x_minus == w_bar,
    w >= 0,
    x_plus >= 0,
    x_minus >= 0,
    w <= U_i,
    cp.sum(cp.abs(w - w_bar)) <= U_TRN,
    phi == cp.sum(c_B * x_plus + c_S * x_minus)
]

# Solve the optimization problem
problem = cp.Problem(objective, constraints)
result = problem.solve()

# Get the optimal portfolio weights
optimal_weights = w.value
