import unittest
from quqfin.core.market_making.orderbook import OrderBook

class TestOrderBook(unittest.TestCase):
    def setUp(self):
        self.orderbook = OrderBook()
        # Initialize with some orders
        self.orderbook.add_bid(100.0, 1000)
        self.orderbook.add_bid(99.5, 2000)
        self.orderbook.add_ask(101.0, 1500)
        self.orderbook.add_ask(101.5, 2500)
    
    def test_best_bid(self):
        self.assertEqual(self.orderbook.get_best_bid(), 100.0)
    
    def test_best_ask(self):
        self.assertEqual(self.orderbook.get_best_ask(), 101.0)
    
    def test_spread(self):
        spread = self.orderbook.get_spread()
        self.assertEqual(spread, 1.0)
    
    def test_empty_orderbook(self):
        empty_ob = OrderBook()
        self.assertIsNone(empty_ob.get_best_bid())
        self.assertIsNone(empty_ob.get_best_ask())
        self.assertIsNone(empty_ob.get_spread())
    
    def test_volume_aggregation(self):
        self.orderbook.add_bid(100.0, 500)  # Add to existing price level
        bids = self.orderbook.bids
        self.assertEqual(bids[100.0], 1500)  # Should be 1000 + 500
    
    def test_price_priority(self):
        self.orderbook.add_bid(100.5, 800)  # Better price
        self.assertEqual(self.orderbook.get_best_bid(), 100.5)

if __name__ == '__main__':
    unittest.main()