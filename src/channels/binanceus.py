from binance import Client
from pprint import pprint
from datetime import datetime, timedelta
import pandas as pd
import random

DTFORMAT = "%Y-%m-%dT%H:%M:%SZ"

import logging

logging.basicConfig(
    format="[%(name)s] %(levelname)s %(asctime)s %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


class BinanceUS:
    def __init__(
        self,
        api_key=open("keys/binanceus-public").read().strip(),
        api_secret=open("keys/binanceus-private").read().strip(),
    ) -> None:
        self.client = Client(api_key, api_secret, tld="us")
        self.availables = [
            x["symbol"] for x in self.client.get_all_tickers()
        ]  # available symbols

    def get_all_tickers(self):
        return self.client.get_all_tickers()

    def get_ticker(self, ticker):
        """Get the latest price for a ticker"""
        try:
            return self.client.get_symbol_ticker(symbol=ticker)["price"]
        except:
            return -1

    def get_bars(
        self,
        ticker,
        interval=Client.KLINE_INTERVAL_1MINUTE,
        start_time=datetime.now() - timedelta(hours=24),
        end_time=datetime.now(),
    ):
        """Get the bars (historical) for ticker
        Params
        ------
        ticker : str
            ticker to search for
        timeframe : ? = Client.KLINE_INTERVAL_1MINUTE
            the timeframe
        start_time : timedelta = datetime.now() - timedelta(hours=24),
        end_time : timedelta = datetime.now()

        Returns
        -------
        df : pd.DataFrame
        """
        klines = self.client.get_historical_klines(
            ticker, interval, start_time.strftime(DTFORMAT), end_time.strftime(DTFORMAT)
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
        logger.info(f"Fetched bars for '{ticker}'")
        return df


if __name__ == "__main__":
    btest = BinanceUS()
    btest.get_bars("BTCUSD")
    tickers = [
        "BTCUSD",
        "ETHUSD",
        "DASHUSD",
        "BCHUSD",
        "LTCUSD",
        "XTZUSD",
    ]
    random.shuffle(tickers)

    bowler = Bowl(window=15, sigma=1.85)
    # bowler = MeanReversion()
    for ticker in tickers:
        print(f"[{ticker}]")
        # bowler.optimize(df["close"])
        df = btest.get_bars(ticker)

        bowler.backtest(df["close"], loud=True)
