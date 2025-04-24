import cupy as cp
from cuquantum import cuda
import numpy as np

class MonteCarloOption:
    def __init__(self, n_simulations=100000, n_steps=252):
        self.n_simulations = n_simulations
        self.n_steps = n_steps
    
    def simulate_paths(self, S0, r, T, sigma):
        dt = T/self.n_steps
        Z = cp.random.standard_normal((self.n_simulations, self.n_steps))
        S = cp.zeros((self.n_simulations, self.n_steps + 1))
        S[:, 0] = S0
        
        for t in range(1, self.n_steps + 1):
            S[:, t] = S[:, t-1] * cp.exp((r - 0.5*sigma**2)*dt + 
                                       sigma*cp.sqrt(dt)*Z[:, t-1])
        return S
    
    def asian_call(self, S0, K, r, T, sigma):
        paths = self.simulate_paths(S0, r, T, sigma)
        avg_prices = cp.mean(paths, axis=1)
        payoff = cp.maximum(avg_prices - K, 0)
        price = float(cp.exp(-r*T) * cp.mean(payoff))
        return price
    
    def lookback_call(self, S0, K, r, T, sigma):
        paths = self.simulate_paths(S0, r, T, sigma)
        max_prices = cp.max(paths, axis=1)
        payoff = cp.maximum(max_prices - K, 0)
        price = float(cp.exp(-r*T) * cp.mean(payoff))
        return price
    
    def barrier_call(self, S0, K, H, r, T, sigma, barrier_type='up-and-out'):
        paths = self.simulate_paths(S0, r, T, sigma)
        max_prices = cp.max(paths, axis=1)
        
        if barrier_type == 'up-and-out':
            valid_paths = max_prices < H
            payoff = cp.where(valid_paths, 
                            cp.maximum(paths[:, -1] - K, 0),
                            0)
        else:  # down-and-out
            valid_paths = cp.min(paths, axis=1) > H
            payoff = cp.where(valid_paths, 
                            cp.maximum(paths[:, -1] - K, 0),
                            0)
            
        price = float(cp.exp(-r*T) * cp.mean(payoff))
        return price
    
    def european_call(self, S0, K, r, T, sigma):
        # Generate random paths using cupy for GPU acceleration
        Z = cp.random.standard_normal(self.n_simulations)
        ST = S0 * cp.exp((r - 0.5*sigma**2)*T + sigma*cp.sqrt(T)*Z)
        payoff = cp.maximum(ST - K, 0)
        price = cp.exp(-r*T) * cp.mean(payoff)
        return float(price)
    
    def european_put(self, S0, K, r, T, sigma):
        Z = cp.random.standard_normal(self.n_simulations)
        ST = S0 * cp.exp((r - 0.5*sigma**2)*T + sigma*cp.sqrt(T)*Z)
        payoff = cp.maximum(K - ST, 0)
        price = cp.exp(-r*T) * cp.mean(payoff)
        return float(price)