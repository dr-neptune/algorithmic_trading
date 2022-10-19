from sys import platform
import tpqoa

api = tpqoa.tpqoa('pyalgo.cfg')

# retrieving historical data
api.get_instruments()

instrument, start, end, granularity, price = 'EUR_USD', '2020-08-10', '2020-08-12', 'M1', 'M'

data = api.get_history(instrument, start, end, granularity, price)

data.info()

import numpy as np
data['returns'] = np.log(data['c'] / data['c'].shift(1))
cols = []

for momentum in [15, 30, 60, 120]:
    col = f'position_{momentum}'
    data[col] = np.sign(data['returns'].rolling(momentum).mean())
    cols.append(col)

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('tkAgg')

strats = ['returns']

for col in cols:
    strat = f"strategy_{col.split('_')[1]}"
    data[strat] = data[col].shift(1) * data['returns']
    strats.append(strat)

data[strats].dropna().cumsum().apply(np.exp).plot()
plt.show()

# Factoring in Leverage and Margin
data[strats].dropna().cumsum().apply(lambda x: x * 20).apply(np.exp).plot()
plt.show()

# Working with streaming data
# doesn't work :/
instrument = 'EUR_USD'
api.stream_data(instrument, stop=10)

# placing market orders
help(api.create_order)

api.create_order(instrument, 10)   # open a long position
api.create_order(instrument, -10)  # goes short after closing the long position via market order

# implementing trading strategies in real time
import tpqoa
import numpy as np
import pandas as pd

class MomentumTrader(tpqoa.tpqoa):
    def __init__(self, conf_file, instrument, bar_length, momentum, units, *args, **kwargs):
        super(MomentumTrader, self).__init__(conf_file)
        self.position = 0
        self.instrument = instrument
        self.momentum = momentum
        self.bar_length = bar_length
        self.units = units
        self.raw_data = pd.DataFrame()
        self.min_length = self.momentum + 1

    def on_success(self, time, bid, ask):
        """takes actions when new tick data arrives"""
        print(self.ticks, end=' ')
        self.raw_data = self.raw_data.append(pd.DataFrame({'bid': bid, 'ask': ask},
                                                          index=[pd.Timestamp(time)]))
        self.data = self.raw_data.resample(self.bar_length, label='right').last().ffill().iloc[:-1]
        self.data['mid'] = self.data.mean(axis=1)
