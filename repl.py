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
