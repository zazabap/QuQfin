import unittest
import numpy as np
from quqfin.core.portfolio.optimization import PortfolioOptimizer

class TestPortfolioOptimizer(unittest.TestCase):
    def setUp(self):
        self.returns = np.array([0.1, 0.15, 0.12, 0.09])
        self.cov_matrix = np.array([[0.1, 0.02, 0.03, 0.02],
                                  [0.02, 0.15, 0.03, 0.02],
                                  [0.03, 0.03, 0.12, 0.02],
                                  [0.02, 0.02, 0.02, 0.1]])
        self.optimizer = PortfolioOptimizer(self.returns, self.cov_matrix)
    
    def test_markowitz_weights_sum(self):
        weights = self.optimizer.markowitz_optimization(target_return=0.12)
        self.assertAlmostEqual(np.sum(weights), 1.0)
    
    def test_risk_parity_weights_sum(self):
        weights = self.optimizer.risk_parity_optimization()
        self.assertAlmostEqual(np.sum(weights), 1.0)
    
    def test_weights_bounds(self):
        weights = self.optimizer.markowitz_optimization(target_return=0.12)
        self.assertTrue(all(0 <= w <= 1 for w in weights))

if __name__ == '__main__':
    unittest.main()