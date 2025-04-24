import unittest
import numpy as np
from quqfin.core.risk.var import RiskMetrics

class TestRiskMetrics(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.returns = np.random.normal(0.001, 0.02, 1000)
        self.risk_metrics = RiskMetrics(self.returns)
    
    def test_var_bounds(self):
        var_95 = self.risk_metrics.value_at_risk(0.95)
        self.assertTrue(var_95 < 0)  # VaR should be negative for losses
        
    def test_cvar_consistency(self):
        var_95 = self.risk_metrics.value_at_risk(0.95)
        cvar_95 = self.risk_metrics.conditional_var(0.95)
        self.assertTrue(cvar_95 <= var_95)  # CVaR should be more conservative
    
    def test_sharpe_ratio(self):
        sharpe = self.risk_metrics.sharpe_ratio(risk_free_rate=0.02)
        self.assertIsInstance(sharpe, float)

if __name__ == '__main__':
    unittest.main()