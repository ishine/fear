from channels.binanceus import BinanceUS
from pprint import pformat, pprint
from strategies_support.predictors import FEDNNPredictor
from strategies_support.cyclers import BaseCycler
from channels.alpaca import Alpaca
from db.tables import Bar, Quote
from db.db import *
import logging
import pandas as pd
from datetime import datetime
from dotmap import DotMap
import pytz
from multiprocessing import Pool, Process
import threading
from abraham3k.prophets import Abraham
from time import sleep

DTFORMAT = "%Y-%m-%dT%H:%M:%SZ"


logging.basicConfig(
    # filename,
    format="[%(name)s] %(levelname)s %(asctime)s %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class FEARv1Alpaca(BaseCycler):
    def __init__(self) -> None:
        super().__init__()
        self.alpaca = Alpaca()

    async def quote_callback(self, q):
        """Log and make a decision based on the price"""
        logging.info(q)
        nq = Quote(q)

    def cycle_trades(self, symbol):
        """Stream data"""
        self.alpaca.stream.subscribe_quotes(self.quote_callback, symbol)
        self.alpaca.stream.run()


class FEARv1Binance(BaseCycler):
    def __init__(
        self, connection, username, password, host, port, database, reset=False
    ) -> None:
        super().__init__(connection, username, password, host, port, database)
        self.api = BinanceUS()
        self.bar_track = pd.DataFrame(
            columns=[
                "datetime",
                "symbol",
                "opn",
                "close",
                "low",
                "high",
                "volume",
                "open_time",
                "close_time",
                "num_trades",
            ]
        )  # for tracking
        if reset:
            reset_tables(self.engine)

    def append_bar_track(self, bar: Bar):
        """Append a bar object to bar_track and database"""
        row = [
            bar.datetime,
            bar.symbol,
            bar.opn,
            bar.close,
            bar.low,
            bar.high,
            bar.volume,
            bar.open_time,
            bar.close_time,
            bar.num_trades,
        ]
        self.bar_track.loc[self.bar_track.shape[0]] = row
        add(self.engine, bar)

    def create_bar(self, raw_bar):
        """Create bar obj from bar msg"""
        data = DotMap(raw_bar)
        bar = data.k

        # timestamps are in ms so convert to s and then to datetime
        bar.t = datetime.fromtimestamp(bar.t / 1000)
        bar.T = datetime.fromtimestamp(bar.T / 1000)
        data.E = datetime.fromtimestamp(data.E / 1000)

        bar_obj = Bar(
            symbol=bar.s,
            opn=bar.o,
            close=bar.c,
            low=bar.l,
            high=bar.h,
            volume=bar.q,
            datetime=data.E,
            open_time=bar.t,
            close_time=bar.T,
            num_trades=bar.n,
        )  # for database
        return bar_obj

    def on_quote(self, msg):
        """Callback"""
        bar = self.create_bar(msg)
        logger.info(bar)
        self.append_bar_track(bar)

    def cycle_trades(
        self,
        symbol: str,
    ):
        """Stream data"""
        logger.info(f"Start stream for {symbol}")
        # start is required to initialise its internal loop
        self.api.stream.start()
        self.api.stream.start_kline_socket(callback=self.on_quote, symbol=symbol)
        self.api.stream.join()  # block


if __name__ == "__main__":
    fv1 = FEARv1Binance("mysql", "test", "test", "localhost", "3306", "fear")
    fv1.cycle("BTCUSDT", "bitcoin good")