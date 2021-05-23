from channels.alpaca import Alpaca, TimeFrame
from dnn import FEDNN
import threading
import pandas as pd
from time import sleep
from datetime import datetime, timedelta
import logging, math

logger = logging.getLogger(__name__)


class Cycler:
    def __init__(
        self,
        api_key: str = open("keys/alpaca_paper_public").read().strip(),
        api_secret: str = open("keys/alpaca_paper_private").read().strip(),
        base_url: str = "https://paper-api.alpaca.markets",
    ) -> None:
        self.fednn = FEDNN()
        self.alpaca = Alpaca(api_key, api_secret, base_url)

    def trade(
        self, ticker: str, side: str, price: float, qty: int, open_blocker: bool = True
    ):
        """Trades if all checks pass"""
        buying_power = self.alpaca.get_buying_power()
        num_shares = self.alpaca.get_shares(ticker)
        open_trades = self.alpaca.any_open_orders()

        if (not open_trades) or open_blocker:
            if side == "buy":
                if buying_power >= price * qty:
                    self.alpaca.submit_limit_order(
                        ticker=ticker, side=side, price=price, qty=qty
                    )
                    return True
                else:
                    logger.warning(
                        f"Not enough balance to buy ({buying_power} < {price*qty})"
                    )
            elif side == "sell":
                if num_shares >= qty:
                    self.alpaca.submit_limit_order(
                        ticker=ticker, side=side, price=price, qty=qty
                    )
                    return True
                else:
                    logger.warning(f"Not enough shares to sell ({num_shares} < {qty})")
            elif side == "hold":  # do nothing
                return True
        else:
            logger.warning("Open trades... not trading")
            return False
        return False

    def cycle_trades(self, ticker: str, spend_amt: float = 1000):
        """
        Cycles between predicting and trading

        Params
        ------
        ticker : str
            stock ticker
        spend_amt : int = 1000
            max amount of money to spend
        """
        # sleep(60 - datetime.now().second)  # sleep til next min starts
        while True:
            # get the data
            data = self.alpaca.get_bars(ticker)
            close = data.close
            price = close[-1]
            qty = math.floor(spend_amt / price)
            signal = self.fednn.get_signal(data)
            # act
            self.trade(ticker, signal, price, qty)
            # sleep til next min start
            sleep(60 - datetime.now().second)

    def cycle_train(self, ticker: str, train_interval: int = 30):
        """
        Cycles training the prediction model

        Params
        ------
        ticker : str
            stock ticker
        train_interval : int
            how many minutes between trains
        """
        # model must have been built first
        interval = 60 * train_interval  # convert to seconds
        while True:
            data = self.alpaca.get_bars(
                ticker,
                timeframe=TimeFrame.Minute,
                start_time=datetime.now() - timedelta(days=14),
            )
            self.fednn.train(data)
            sleep(interval)

    def cycle(ticker: str, spend_amt: float, train_interval: int = 30):
        """
        first train then start
        """
