import unittest
import numpy as np
from quqfin.core.algo_trading.strategies import TradingStrategies

class TestTradingStrategies(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        # Simulate trending price series
        trend = np.linspace(0, 0.5, 1000)
        noise = np.random.normal(0, 0.02, 1000)
        self.prices = 100 * np.exp(trend + noise)
        self.trader = TradingStrategies(self.prices)
    
    def test_ma_crossover_signals(self):
        signals = self.trader.moving_average_crossover(short_window=20, long_window=50)
        self.assertEqual(len(signals), len(self.prices))
        self.assertTrue(all(s in [-1, 0, 1] for s in signals))
    
    def test_momentum_strategy(self):
        signals = self.trader.momentum_strategy(lookback=20)
        self.assertEqual(len(signals), len(self.prices))
        self.assertTrue(all(s in [-1, 0, 1] for s in signals))
    
    def test_mean_reversion(self):
        signals = self.trader.mean_reversion(window=20, z_score_threshold=2)
        self.assertEqual(len(signals), len(self.prices))
        self.assertTrue(all(s in [-1, 0, 1] for s in signals))
    
    def test_trending_market_momentum(self):
        # Momentum should detect the trend
        signals = self.trader.momentum_strategy(lookback=50)
        # Check if majority of signals are positive in trending market
        positive_signals = np.sum(signals == 1)
        self.assertTrue(positive_signals > len(signals) * 0.4)

if __name__ == '__main__':
    unittest.main()