'''
Script for processing SPY.csv for feeding to an lstm model
'''

import pandas as pd
import talib

spy = pd.read_csv("RealTime.csv")

# simple moving averages
spy['20_sma'] = spy['Close'].rolling(window=20).mean()

# Bollinger bands
spy['stddev'] = spy['Close'].rolling(window=20).std()
spy['upper_bb'] = spy['20_sma'] + (2 * spy['stddev'])
spy['lower_bb'] = spy['20_sma'] - (2 * spy['stddev'])

# RSI 14
RSI_PERIOD = 14
spy['rsi'] = talib.RSI(spy['Close'], RSI_PERIOD)

# MACD
MACD_FAST_EMA = 12
MACD_SLOW_EMA = 26
MACD_SIGNAL_PERIOD = 9
macd, signal, hist = talib.MACD(spy['Close'], 
                                fastperiod=MACD_FAST_EMA, slowperiod=MACD_SLOW_EMA, signalperiod=MACD_SIGNAL_PERIOD)
spy = pd.concat([spy, macd, signal, hist], axis=1)
spy = spy.rename(columns={0:'macd',1:'signal',2:'hist'})

dataset = spy.filter(['Volume', '20_sma', 'upper_bb', 'lower_bb', 'rsi', 'macd', 'signal', 'hist','Close'])
dataset = dataset.iloc[33:,:]
dataset['targetNextClose'] = dataset['Close'].shift(-1)
dataset['target'] = dataset['targetNextClose']-dataset['Close']
dataset.dropna()
dataset = dataset.iloc[:-1,:]
dataset.to_csv("real_dataset.csv",index=False)