import numpy as np
from collections import defaultdict

class OrderBook:
    def __init__(self):
        self.bids = defaultdict(float)
        self.asks = defaultdict(float)
        
    def add_bid(self, price, volume):
        self.bids[price] += volume
        
    def add_ask(self, price, volume):
        self.asks[price] += volume
        
    def get_best_bid(self):
        if not self.bids:
            return None
        return max(self.bids.keys())
    
    def get_best_ask(self):
        if not self.asks:
            return None
        return min(self.asks.keys())
    
    def get_spread(self):
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        if best_bid is None or best_ask is None:
            return None
        return best_ask - best_bid