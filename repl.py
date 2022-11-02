import math
import time
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('tkAgg')
np.random.seed(50)

# simulate 50 series with 100 coin tosses per series
p = 0.55                 # fix probability for heads
f = p - (1 - p)          # calculate optimal fraction
I = 50                   # number of series to calc
n = 100                  # number of trials per series

def run_simulation(f):
    c = np.zeros((n, I))                               # store results
    c[0] = 100                                         # start with 100
    for i in range(I):                                 # simulations
        for t in range(1, n):                          # build series
            o = np.random.binomial(1, p)               # toss a coin
            if o > 0:                                  # heads
                c[t, i] = (1 + f) * c[t - 1, i]        # add to win capital
            else:                                      # tails
                c[t, i] = (1 - f) * c[t - 1, i]        # subtract from win capital
    return c

c1 = run_simulation(f)
c1.round(2)


fig = plt.figure()
plt.plot(c1, 'b', lw=0.5)
plt.plot(c1.mean(axis=1), 'r', lw=2.5)
plt.show()

# try different values
c2 = run_simulation(0.05)
c3 = run_simulation(0.25)
c4 = run_simulation(0.5)

# average capital over time for different values of f
fig = plt.figure()
plt.plot(c1.mean(axis=1), 'r', label='$f^*=0.1$')
plt.plot(c2.mean(axis=1), 'b', label='$f^*=0.05$')
plt.plot(c3.mean(axis=1), 'y', label='$f^*=0.25$')
plt.plot(c4.mean(axis=1), 'm', label='$f^*=0.5$')
plt.legend(loc=0)
plt.show()

raw = pd.read_csv('https://hilpisch.com/pyalgo_eikon_eod_data.csv',
                  index_col=0, parse_dates=True)

symbol = '.SPX'

data = pd.DataFrame(raw[symbol])
data['return'] = np.log(data / data.shift(1))
data = data.dropna()
data.tail()

mu = data['return'].mean() * 252
sigma = data['return'].std() * 252 ** 0.5
r = 0.0
f = (mu - r) / sigma ** 2
f

from typing import List

def kelly_strategy(data: pd.DataFrame, strategies: List[float]):
    for strategy in strategies:
        equ = f'equity_{strategy:.2f}'
        cap = f'capital_{strategy:.2f}'
        data[equ] = 1
        data[cap] = data[equ] * strategy
        for i, t in enumerate(data.index[1:]):
            t1 = data.index[i]
            data.loc[t, cap] = data[cap].loc[t1] * math.exp(data['return'].loc[t])
            data.loc[t, equ] = (data[cap].loc[t] -
                                data[cap].loc[t1] +
                                data[equ].loc[t1])
            data.loc[t, cap] = data[equ].loc[t] * f
    return data



output = kelly_strategy(data, [f * 0.5, f * 0.66, f])
output.tail().iloc[0]


equs = []
def kelly_strategy(f):
    global equs
    equ = 'equity_{:.2f}'.format(f)
    equs.append(equ)
    cap = 'capital_{:.2f}'.format(f)
    data[equ] = 1
    data[cap] = data[equ] * f
    for i, t in enumerate(data.index[1:]):
        t_1 = data.index[i]
        data.loc[t, cap] = data[cap].loc[t_1] * \
          math.exp(data['return'].loc[t])
        data.loc[t, equ] = data[cap].loc[t] - \
          data[cap].loc[t_1] + \
          data[equ].loc[t_1]
        data.loc[t, cap] = data[equ].loc[t] * f

kelly_strategy(f * 0.5)
kelly_strategy(f * 0.55)
kelly_strategy(f)
print(data[equs].tail())

ax = data['return'].cumsum().apply(np.exp).plot()
data[equs].plot(ax=ax, legend=True)
plt.show()
