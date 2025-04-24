import unittest
import numpy as np
from quqfin.core.timeseries.models import TimeSeriesModels

class TestTimeSeriesModels(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.prices = 100 * np.exp(np.random.normal(0.001, 0.02, 1000).cumsum())
        self.ts_model = TimeSeriesModels(self.prices)
    
    def test_garch_positive_volatility(self):
        returns = np.diff(np.log(self.prices))
        volatility = self.ts_model.garch(returns)
        self.assertTrue(all(v > 0 for v in volatility))
    
    def test_kalman_filter_shape(self):
        observations = np.diff(np.log(self.prices))
        state_mean, state_cov = self.ts_model.kalman_filter(
            observations, 1, 1, 0.01, 0.1)
        self.assertEqual(len(state_mean), len(observations))

if __name__ == '__main__':
    unittest.main()