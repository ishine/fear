from channels.oanda import OANDA
import unittest
from datetime import datetime, timedelta

from channels.alpaca import Alpaca, TimeFrame
from channels.binanceus import BinanceUS
from channels.screener import Screener
from strategies.base import BaseCycler, BaseStrategy
from strategies.dnn import FEDNNStrategy
from strategies.knearest import FEKNNStrategy


class TestScreener(unittest.TestCase):
    def test_human_to_number(self):
        cs = "1.7M"
        self.assertEquals(Screener()._human_to_number(cs), 1700000)

    def test_active(self):
        received = Screener().get_active(count=10)
        self.assertFalse(received.empty)

    def test_losers(self):
        received = Screener().get_losers(count=10)
        self.assertFalse(received.empty)

    def test_gainers(self):
        received = Screener().get_gainers(count=10)
        self.assertFalse(received.empty)

    def test_trending(self):
        received = Screener().get_trending(count=10)
        self.assertFalse(received.empty)


class TestAlpaca(unittest.TestCase):
    def test_get_bars(self):
        symbol = "gld"
        received = Alpaca().get_bars(
            symbol,
            timeframe=TimeFrame.Day,
            start_time=datetime.now() - timedelta(weeks=52),
        )
        self.assertFalse(received.empty)
        received = Alpaca().get_bars(
            symbol,
            timeframe=TimeFrame.Day,
            start_time=datetime.now() - timedelta(weeks=52),
            resample=5,
        )
        self.assertFalse(received.empty)

    def test_import_stock_data(self):
        symbols = ["gld", "spy"]
        received = Alpaca().import_stock_data(
            symbols,
            start_time=datetime.now() - timedelta(weeks=52),
        )
        self.assertFalse(received.empty)

    def test_get_account(self):
        received = Alpaca().get_account()
        self.assertIsNotNone(received)

    def test_any_open_orders(self):
        received = Alpaca().any_open_orders()
        self.assertTrue(type(received) is bool)

    def test_get_shares(self):
        symbol = "gld"
        received = Alpaca().get_shares(symbol)
        self.assertTrue(type(received) is float or type(received) is int)

    def test_get_buying_power(self):
        received = Alpaca().get_buying_power()
        self.assertTrue(type(received) is float or type(received) is int)


class TestOANDA(unittest.TestCase):
    def test_get_bars(self):
        symbol = "EUR_USD"
        received = OANDA().get_bars(
            symbol,
            timeframe="M1",
            start_time=datetime.now() - timedelta(days=2),
        )
        self.assertFalse(received.empty)

    def test_get_symbols(self):
        received = OANDA().get_symbols()
        self.assertTrue(received)


if __name__ == "__main__":
    unittest.main()
