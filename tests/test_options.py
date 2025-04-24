import unittest
import numpy as np
from quqfin.core.options.black_scholes import BlackScholes
from quqfin.core.options.monte_carlo import MonteCarloOption

class TestBlackScholes(unittest.TestCase):
    def setUp(self):
        self.bs = BlackScholes()
        self.S = 100
        self.K = 100
        self.r = 0.05
        self.sigma = 0.2
        self.T = 1
        
    def test_call_put_parity(self):
        call = self.bs.call_price(self.S, self.K, self.r, self.sigma, self.T)
        put = self.bs.put_price(self.S, self.K, self.r, self.sigma, self.T)
        parity = call - put - (self.S - self.K * np.exp(-self.r * self.T))
        self.assertAlmostEqual(parity, 0, places=6)
    
    def test_delta_bounds(self):
        delta = self.bs.delta_call(self.S, self.K, self.r, self.sigma, self.T)
        self.assertTrue(0 <= delta <= 1)

class TestMonteCarloOption(unittest.TestCase):
    def setUp(self):
        self.mc = MonteCarloOption(n_simulations=10000)
        self.S = 100
        self.K = 100
        self.r = 0.05
        self.sigma = 0.2
        self.T = 1
        
    def test_barrier_option_bounds(self):
        H = 120
        barrier_price = self.mc.barrier_call(self.S, self.K, H, self.r, self.T, self.sigma)
        vanilla_price = BlackScholes.call_price(self.S, self.K, self.r, self.sigma, self.T)
        self.assertTrue(barrier_price <= vanilla_price)

if __name__ == '__main__':
    unittest.main()