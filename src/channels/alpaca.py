from os import stat
from re import S, search
from alpaca_trade_api.rest import *
from alpaca_trade_api.stream import *
from datetime import date, datetime, timedelta
from time import sleep
from abraham3k import Abraham
import pandas as pd
from multiprocessing import Pool, Process

try:
    from channels.base import BaseAPI
except:
    from base import BaseAPI
import logging

logger = logging.getLogger(__name__)

# Data viz
import plotly.graph_objs as go

DTFORMAT = "%Y-%m-%dT%H:%M:%SZ"


class Alpaca(BaseAPI):
    def __init__(
        self,
        api_key=open("keys/alpaca_paper_public").read().strip(),
        api_secret=open("keys/alpaca_paper_private").read().strip(),
        base_url="https://paper-api.alpaca.markets",
    ) -> None:
        self.api = REST(
            key_id=api_key, secret_key=api_secret, base_url=base_url, api_version="v2"
        )
        self.stream = Stream(key_id=api_key, secret_key=api_secret, base_url=base_url)

    def get_bars(
        self,
        symbol,
        timeframe=TimeFrame.Minute,
        start_time=datetime.now() - timedelta(hours=24),
        end_time=datetime.now(),
        resample: int = 1,
    ):
        """Wrapper for the api to simplify the calls

        Params
        ------
        symbol : str
            symbol to search for
        timeframe : TimeFrame = TimeFrame.Minute
            the timeframe
        start_time : timedelta = datetime.now() - timedelta(hours=24),
        end_time : timedelta = datetime.now()
        resample : int = 1
            this slices it every n rows

        Returns
        -------
        df : pd.DataFrame
        """
        start = start_time.strftime(DTFORMAT)
        end = end_time.strftime(DTFORMAT)

        try:
            df = self.api.get_bars(
                symbol,
                timeframe,
                start,
                end,
                adjustment="raw",
            ).df
            logger.info(
                f"Fetched {df.shape[0]} bars for '{symbol}' from {start} to {end} with freq {timeframe} and resample {resample}"
            )
            if resample > 1:
                df = df.iloc[df.shape[0] % resample - 1 :: resample]
            return df
        except Exception as e:
            logger.warning(
                f"Couldn't get bars for {symbol} from {start} to {end} with freq {timeframe} ({e})"
            )
            return pd.DataFrame()

    def import_stock_data(
        self,
        symbols: list,
        start_time: timedelta = datetime.now() - timedelta(days=365),
        end_time=datetime.now(),
        interval: timedelta = TimeFrame.Day,
    ):
        """Import stock data and get the close value
        Takes a list and can get multiple

        Params
        ------
        symbol : str
            symbol to search for
        timeframe : TimeFrame = TimeFrame.Minute
            the timeframe
        start_time : timedelta = datetime.now() - timedelta(hours=24),
        end_time : timedelta = datetime.now()

        Returns
        -------
        data : pd.DataFrame
        """
        data = pd.DataFrame()
        if type(symbols) == list:
            for symbol in symbols:
                try:
                    data[symbol] = self.get_bars(
                        symbol,
                        timeframe=interval,
                        start_time=start_time,
                        end_time=end_time,
                    )["close"]
                except Exception as e:
                    logger.warning(f"Unknown error occured while fetching symbols: {e}")
        else:
            logger.warning(f"Input '{symbols}' is not a list of symbols.")
        return data

    def submit_limit_order(self, symbol, side, price, qty=1, time_in_force="day"):
        """Submit a limit order

        Params
        ------
        symbol : str
            symbol to act on
        side : str
            buy or sell
        price : float
            price to buy at
        qty : int
            how many shares to sell/buy
        time_in_force : str
            expire timne

        Returns
        -------
        True if completed
        """
        try:
            self.api.submit_order(
                symbol=symbol,
                qty=qty,  # fractional shares
                side=side,
                type="limit",
                limit_price=price,
                time_in_force=time_in_force,
            )
            logger.info(
                f"Submitted limit {side} order for {qty} {symbol} @ ${price} (TIF={time_in_force})"
            )
            return True
        except Exception as e:
            logger.warning(
                f"Couldn't submit limit {side} order for {qty} {symbol} @ ${price} ({e})"
            )
            return False

    def submit_limit_buy(self, symbol, price, qty=1, time_in_force="day"):
        """Submit a limit order

        Params
        ------
        symbol : str
            symbol to act on
        price : float
            price to buy at
        qty : int
            how many shares to sell/buy
        time_in_force : str
            expire timne

        Returns
        -------
        True if completed
        """
        side = "buy"
        try:
            self.api.submit_order(
                symbol=symbol,
                qty=qty,  # fractional shares
                side=side,
                type="limit",
                limit_price=price,
                time_in_force=time_in_force,
            )
            logger.info(
                f"Submitted limit {side} order for {qty} {symbol} @ ${price} (TIF={time_in_force})"
            )
            return True
        except Exception as e:
            logger.warning(
                f"Couldn't submit limit {side} order for {qty} {symbol} @ ${price} ({e})"
            )
            return False

    def submit_limit_sell(self, symbol, price, qty=1, time_in_force="day"):
        """Submit a limit order

        Params
        ------
        symbol : str
            symbol to act on
        price : float
            price to buy at
        qty : int
            how many shares to sell/buy
        time_in_force : str
            expire timne

        Returns
        -------
        True if completed
        """
        side = "sell"
        try:
            self.api.submit_order(
                symbol=symbol,
                qty=qty,  # fractional shares
                side=side,
                type="limit",
                limit_price=price,
                time_in_force=time_in_force,
            )
            logger.info(
                f"Submitted limit {side} order for {qty} {symbol} @ ${price} (TIF={time_in_force})"
            )
            return True
        except Exception as e:
            logger.warning(
                f"Couldn't submit limit {side} order for {qty} {symbol} @ ${price} ({e})"
            )
            return False

    def submit_market_order(self, symbol, side, qty=1, time_in_force="day"):
        """Submit a market order

        Params
        ------
        symbol : str
            symbol to act on
        side : str
            buy or sell
        qty : int
            how many shares to sell/buy
        time_in_force : str
            expire timne

        Returns
        -------
        True
        """
        try:
            self.api.submit_order(
                symbol=symbol,
                qty=qty,  # fractional shares
                side=side,
                type="market",
                time_in_force=time_in_force,
            )
            return True
        except Exception as e:
            logger.warning(
                f"Couldn't submit market {side} order for {qty} {symbol} ({e})"
            )
            return False

    def get_account(self):
        """
        Get account info
        """
        try:
            return self.api.get_account()
        except Exception as e:
            logger.warning(f"Couldn't get account info ({e})")
            return None

    def any_open_orders(self) -> bool:
        """
        Returns true if any open orders
        """
        try:
            resp = self.api.list_orders(status="open")
            return len(resp) > 0
        except Exception as e:
            logger.warning(f"Couldn't get open orders ({e})")
            return True  # better safe than sorry

    def get_position(self, symbol):
        """Gets the position for a symbol"""
        try:
            return self.api.get_position(symbol)
        except Exception as e:
            logger.warning(f"Couldn't get position for {symbol} ({e})")
            return {}

    def get_shares(self, symbol):
        """Returns the share count that you possess for a symbol"""
        try:
            return self.api.get_position(symbol)["qty"]
        except Exception as e:  # effectively 0 then
            return 0

    def get_buying_power(self):
        """Returns the buying_power you have"""
        try:
            return float(self.api.get_account().buying_power)
        except Exception as e:
            logging.warning(f"Couldnt get buying_power ({e})")
            return 0


if __name__ == "__main__":
    ape = Alpaca()
    print(
        ape.get_bars("tsla", start_time=datetime.now() - timedelta(days=10), resample=5)
    )