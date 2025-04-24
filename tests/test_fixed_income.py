import unittest
from quqfin.core.fixed_income.bonds import BondPricing

class TestBondPricing(unittest.TestCase):
    def setUp(self):
        self.bond = BondPricing()
        self.face_value = 1000
        self.coupon_rate = 0.05
        self.yield_rate = 0.05
        self.maturity = 2
    
    def test_zero_coupon_price(self):
        price = self.bond.zero_coupon_price(self.face_value, self.yield_rate, self.maturity)
        self.assertTrue(price < self.face_value)
    
    def test_par_bond(self):
        # When coupon rate equals yield, price should equal face value
        price = self.bond.coupon_bond_price(self.face_value, self.coupon_rate, 
                                          self.coupon_rate, self.maturity)
        self.assertAlmostEqual(price, self.face_value, places=2)
    
    def test_duration_positive(self):
        duration = self.bond.duration(self.face_value, self.coupon_rate, 
                                    self.yield_rate, self.maturity)
        self.assertTrue(duration > 0)

if __name__ == '__main__':
    unittest.main()