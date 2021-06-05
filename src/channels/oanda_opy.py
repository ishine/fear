try:
    from channels.base import BaseAPI
except:
    from base import BaseAPI
from os import access
import oandapy, logging, statistics
from datetime import datetime, timedelta
import pandas as pd

logging.basicConfig(
    # filename,
    format="[%(name)s] %(levelname)s %(asctime)s %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

DTFORMAT = "%Y-%m-%dT%H:%M:%SZ"


class OANDAOPYStream(oandapy.Streamer):
    def __init__(self, count=10, *args, **kwargs):
        super(OANDAOPYStream, self).__init__(*args, **kwargs)
        self.count = count
        self.reccnt = 0

    def on_success(self, data):
        print(data)
        self.reccnt += 1
        if self.reccnt == self.count:
            self.disconnect()

    def on_error(self, data):
        self.disconnect()


class OANDA(BaseAPI):
    def __init__(
        self,
        account_id: str = open("keys/oanda-id").read().strip(),
        access_token: str = open("keys/oanda-access-token").read().strip(),
        mode: str = "practice",
    ) -> None:
        self.account_id = account_id
        self.api = oandapy.API(environment=mode, access_token=access_token)
        self.streamer = OANDAOPYStream(environment=mode, access_token=access_token)

    def get_symbols(self):
        """Get available symbols"""
        try:
            return self.api.get_instruments(
                account_id=self.account_id,
            )
        except Exception as e:
            logging.warning(f"Couldn't get instruments ({e})")
            return []

    def get_bars(
        self,
        ticker,
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
        ticker : str
            ticker to search for
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
        start = start_time.replace(second=0, microsecond=0).strftime(DTFORMAT)
        end = end_time.replace(second=0, microsecond=0).strftime(DTFORMAT)
        params = {
            "instrument": ticker,
            "granularity": timeframe,
            "price": book,
            "from": start,
            "to": end,
            "count": limit,  # we want the max
        }
        try:
            response = self.api.get_history(**params)
            df = pd.DataFrame(response["candles"])
            df.index.rename("timestamp", inplace=True)
            df["close"] = (df["closeBid"] + df["closeAsk"]) / 2

            logger.info(
                f"Fetched {df.shape[0]} bars for '{ticker}' from {start} to {end} with freq {timeframe} and resample {resample}"
            )
            if resample > 1:
                df = df.iloc[df.shape[0] % resample - 1 :: resample]
            return df
        except Exception as e:
            logger.warning(
                f"Couldn't get bars for {ticker} from {start} to {end} with freq {timeframe} ({e})"
            )
            return pd.DataFrame()

    def submit_limit_order(self, ticker, side, price, qty=1, time_in_force="day"):
        """Submit a limit order

        Params
        ------
        ticker : str
            ticker to act on
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
                symbol=ticker,
                qty=qty,  # fractional shares
                side=side,
                type="limit",
                limit_price=price,
                time_in_force=time_in_force,
            )
            logger.info(
                f"Submitted limit {side} order for {qty} {ticker} @ ${price} (TIF={time_in_force})"
            )
            return True
        except Exception as e:
            logger.warning(
                f"Couldn't submit limit {side} order for {qty} {ticker} @ ${price} ({e})"
            )
            return False


if __name__ == "__main__":
    o = OANDA()
    o.streamer.rates(o.account_id, instruments="EUR_USD,EUR_JPY,US30_USD,DE30_EUR")
    print(o.get_bars("EUR_USD"))
