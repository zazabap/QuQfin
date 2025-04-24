import numpy as np
import pandas as pd

class TradingStrategies:
    def __init__(self, prices):
        self.prices = np.array(prices)
        
    def moving_average_crossover(self, short_window=50, long_window=200):
        short_ma = pd.Series(self.prices).rolling(window=short_window).mean()
        long_ma = pd.Series(self.prices).rolling(window=long_window).mean()
        
        signals = np.zeros(len(self.prices))
        signals[short_ma > long_ma] = 1  # Buy signal
        signals[short_ma < long_ma] = -1  # Sell signal
        
        return signals
    
    def momentum_strategy(self, lookback=20):
        returns = np.diff(np.log(self.prices))
        momentum = np.zeros(len(self.prices))
        momentum[lookback:] = np.array([np.sum(returns[i-lookback:i]) 
                                      for i in range(lookback, len(returns))])
        
        signals = np.zeros(len(self.prices))
        signals[momentum > 0] = 1
        signals[momentum < 0] = -1
        
        return signals
    
    def mean_reversion(self, window=20, z_score_threshold=2):
        rolling_mean = pd.Series(self.prices).rolling(window=window).mean()
        rolling_std = pd.Series(self.prices).rolling(window=window).std()
        z_scores = (self.prices - rolling_mean) / rolling_std
        
        signals = np.zeros(len(self.prices))
        signals[z_scores > z_score_threshold] = -1  # Overbought
        signals[z_scores < -z_score_threshold] = 1  # Oversold
        
        return signals