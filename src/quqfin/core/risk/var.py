import numpy as np
import cupy as cp

class RiskMetrics:
    def __init__(self, returns):
        self.returns = cp.array(returns)
    
    def value_at_risk(self, confidence_level=0.95):
        return float(cp.percentile(self.returns, (1 - confidence_level) * 100))
    
    def conditional_var(self, confidence_level=0.95):
        var = self.value_at_risk(confidence_level)
        return float(cp.mean(self.returns[self.returns <= var]))
    
    def sharpe_ratio(self, risk_free_rate=0.0):
        excess_returns = self.returns - risk_free_rate
        return float(cp.mean(excess_returns) / cp.std(excess_returns))