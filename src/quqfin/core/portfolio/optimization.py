import numpy as np
import cupy as cp
from scipy.optimize import minimize

class PortfolioOptimizer:
    def __init__(self, returns, cov_matrix):
        self.returns = cp.array(returns)
        self.cov_matrix = cp.array(cov_matrix)
    
    def risk_parity_optimization(self):
        n_assets = len(self.returns)
        
        def risk_contribution(weights):
            portfolio_vol = cp.sqrt(cp.dot(weights.T, cp.dot(self.cov_matrix, weights)))
            marginal_risk = cp.dot(self.cov_matrix, weights) / portfolio_vol
            risk_contrib = weights * marginal_risk
            return risk_contrib
        
        def objective(weights):
            risk_contrib = risk_contribution(cp.array(weights))
            target_risk = 1.0 / n_assets
            return float(cp.sum((risk_contrib - target_risk)**2))
        
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        bounds = tuple((0, 1) for _ in range(n_assets))
        
        result = minimize(objective,
                        x0=np.ones(n_assets)/n_assets,
                        method='SLSQP',
                        bounds=bounds,
                        constraints=constraints)
        
        return result.x
    
    def black_litterman(self, market_caps, views, view_confidences):
        tau = 0.05  # Prior uncertainty scaling
        pi = self.returns  # Market equilibrium returns
        
        # Compute posterior returns and covariance
        omega = np.diag(1/view_confidences)
        posterior_returns = np.linalg.inv(
            np.linalg.inv(tau * self.cov_matrix) + views.T @ np.linalg.inv(omega) @ views
        ) @ (np.linalg.inv(tau * self.cov_matrix) @ pi + views.T @ np.linalg.inv(omega) @ views)
        
        return self.markowitz_optimization(target_return=float(cp.mean(posterior_returns)))
        
    def markowitz_optimization(self, target_return=None):
        n_assets = len(self.returns)
        
        def objective(weights):
            weights = cp.array(weights)
            portfolio_var = cp.dot(weights.T, cp.dot(self.cov_matrix, weights))
            return float(portfolio_var)
        
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        ]
        
        if target_return is not None:
            constraints.append(
                {'type': 'eq', 
                 'fun': lambda x: float(cp.dot(self.returns, cp.array(x))) - target_return}
            )
            
        bounds = tuple((0, 1) for _ in range(n_assets))
        
        result = minimize(objective, 
                        x0=np.ones(n_assets)/n_assets,
                        method='SLSQP',
                        bounds=bounds,
                        constraints=constraints)
        
        return result.x