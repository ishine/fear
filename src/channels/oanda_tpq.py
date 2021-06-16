try:
    from channels.base import BaseAPI
except:
    from base import BaseAPI
from os import access
import tpqoa, logging, statistics
from datetime import datetime, timedelta
import pandas as pd

logging.basicConfig(
    # filename,
    format="[%(name)s] %(levelname)s %(asctime)s %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

DTFORMAT = "%Y-%m-%dT%H:%M:%SZ"


class OANDA(BaseAPI, tpqoa.tpqoa):
    def __init__(self, config: str = "keys/oanda.cfg") -> None:
        super(OANDA, self).__init__(config)
        self.api = tpqoa.tpqoa(config)

    def get_symbols(self):
        """Get available symbols"""
        try:
            return self.api.get_instruments()
        except Exception as e:
            logging.warning(f"Couldn't get instruments ({e})")
            return []

    def get_bars(
        self,
        symbol,
        timeframe: str = "M1",
        start_time: timedelta = datetime.now() - timedelta(hours=24),
        end_time: timedelta = datetime.now(),
        book: str = "M",
        resample: int = 1,
        limit: int = 2000,
    ):
        """Wrapper for the api to simplify the calls

        Params
        ------
        symbol : str
            symbol to search for
        timeframe : str = "M1"
            the timeframe (granularity) - see
            http://developer.oanda.com/rest-live-v20/instrument-df/#CandlestickGranularity
            for definition
        start_time : timedelta = datetime.now() - timedelta(hours=24),
        end_time : timedelta = datetime.now()
        resample : int = 1
            this slices it every n rows
        book : str = "M"
            "M", "A", "B" - midpoint, ask, bid
        limit : int = 2000
            max count we want to return

        Returns
        -------
        df : pd.DataFrame
        """
        start = start_time.replace(second=0, microsecond=0)
        end = end_time.replace(second=0, microsecond=0)
        try:
            df = self.api.get_history(
                instrument=symbol,
                granularity=timeframe,
                price=book,
                start=start,
                end=end,
            )
            df.index.rename("timestamp", inplace=True)

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

    def submit_market_order(
        self,
        symbol,
        side,
        qty=1,
    ):
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
        True if completed
        """
        if side == "sell":
            multiplier = -1
        else:
            multiplier = 1
        try:
            self.api.submit_order(
                symbol,
                units=qty * multiplier,
            )
            logger.info(f"Submitted market {side} order for {qty} {symbol}")
            return True
        except Exception as e:
            logger.warning(
                f"Couldn't submit market {side} order for {qty} {symbol} ({e})"
            )
            return False

    def on_success(self, time, bid, ask):
        """ Method called when new data is retrieved. """
        logger.info("BID: {:.5f} | ASK: {:.5f}".format(bid, ask))


if __name__ == "__main__":
    o = OANDA()
    o.stream_data("EUR_USD")
    print(o.get_bars("EUR_USD"))
