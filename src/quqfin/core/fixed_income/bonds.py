import numpy as np
import cupy as cp

class BondPricing:
    def __init__(self):
        self.day_count = 360
        
    def zero_coupon_price(self, face_value, r, T):
        return face_value * np.exp(-r * T)
    
    def coupon_bond_price(self, face_value, coupon_rate, r, T, freq=2):
        periods = int(T * freq)
        coupon = face_value * coupon_rate / freq
        times = np.arange(1, periods + 1) / freq
        cash_flows = np.array([coupon] * periods)
        cash_flows[-1] += face_value
        return np.sum(cash_flows * np.exp(-r * times))
    
    def duration(self, face_value, coupon_rate, r, T, freq=2):
        periods = int(T * freq)
        coupon = face_value * coupon_rate / freq
        times = np.arange(1, periods + 1) / freq
        cash_flows = np.array([coupon] * periods)
        cash_flows[-1] += face_value
        pv_flows = cash_flows * np.exp(-r * times)
        return np.sum(times * pv_flows) / np.sum(pv_flows)