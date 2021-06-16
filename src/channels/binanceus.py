try:
    from channels.base import BaseAPI
except:
    from base import BaseAPI

import random, asyncio
from datetime import datetime, timedelta
from twisted.internet import reactor

import pandas as pd
from binance import (
    AsyncClient,
    BinanceSocketManager,
    Client,
    ThreadedWebsocketManager,
    client,
)

DTFORMAT = "%Y-%m-%dT%H:%M:%SZ"

import logging

logger = logging.getLogger(__name__)


class BinanceUS(BaseAPI):
    def __init__(
        self,
        api_key=open("keys/binanceus-public").read().strip(),
        api_secret=open("keys/binanceus-private").read().strip(),
    ) -> None:
        self.client = Client(api_key, api_secret, tld="us")
        self.stream = ThreadedWebsocketManager(api_key, api_secret)

    def get_all_symbols(self):
        return self.client.get_all_symbols()

    def get_symbol(self, symbol):
        """Get the latest price for a symbol"""
        try:
            return self.client.get_symbol_symbol(symbol=symbol)["price"]
        except:
            return -1

    def get_bars(
        self,
        symbol,
        interval=Client.KLINE_INTERVAL_1MINUTE,
        start_time=datetime.now() - timedelta(hours=24),
        end_time=datetime.now(),
    ):
        """Get the bars (historical) for symbol
        Params
        ------
        symbol : str
            symbol to search for
        timeframe : ? = Client.KLINE_INTERVAL_1MINUTE
            the timeframe
        start_time : timedelta = datetime.now() - timedelta(hours=24),
        end_time : timedelta = datetime.now()

        Returns
        -------
        df : pd.DataFrame
        """
        start = start_time.strftime(DTFORMAT)
        end = end_time.strftime(DTFORMAT)

        try:
            klines = self.client.get_historical_klines(
                symbol, interval, start, end
            )  # raw

            # convert to float
            for i in range(len(klines)):
                for j in range(1, len(klines[i])):
                    klines[i][j] = float(klines[i][j])
            # turn into dataframe
            index = "open_time"
            df = (
                pd.DataFrame(
                    klines,
                    columns=[
                        "open_time",
                        "open",
                        "high",
                        "low",
                        "close",
                        "volume",
                        "close_time",
                        "quote_volume",
                        "trades",
                        "taker_buy_base_vol",
                        "taker_buy_quote_vol",
                        "ignore",
                    ],
                )
                .drop("ignore", axis=1)
                .rename(columns={index: "timestamp"})
            )

            df.index = pd.to_datetime(df.timestamp, unit="ms")
            df.drop("timestamp", axis=1, inplace=True)
            df = df[["open", "high", "low", "close", "volume"]]
            logger.info(
                f"Fetched {df.shape[0]} bars for '{symbol}' from {start} to {end}"
            )
            return df
        except Exception as e:
            logger.warning(
                f"Couldn't get bars for {symbol} from {start} to {end} with freq {interval} ({e})"
            )
            return pd.DataFrame()


if __name__ == "__main__":
    btest = BinanceUS()
    btest.get_bars("BTCUSD")
    symbols = [
        "BTCUSD",
        "ETHUSD",
        "DASHUSD",
        "BCHUSD",
        "LTCUSD",
        "XTZUSD",
    ]
    random.shuffle(symbols)

    bowler = Bowl(window=15, sigma=1.85)
    # bowler = MeanReversion()
    for symbol in symbols:
        print(f"[{symbol}]")
        # bowler.optimize(df["close"])
        df = btest.get_bars(symbol)

        bowler.backtest(df["close"], loud=True)
