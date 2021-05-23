# momentum
from channels.alpaca import Alpaca
import pandas as pd, numpy as np
from finta import TA as ta
import logging, os
from tqdm import trange, tqdm

os.environ["NUMEXPR_MAX_THREADS"] = "8"

logging.basicConfig(
    format="[%(name)s] %(levelname)s %(asctime)s %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


class Momentum:
    """Momentum trading
    Workflow:
    Initialize
    """

    def __init__(
        self,
        price_ma: int = 30,
        volume_ma: int = 30,
        dfcols: dict = {
            "close": "close",
            "open": "open",
            "low": "low",
            "high": "high",
            "volume": "volume",
        },
    ) -> None:
        self.close = dfcols["close"]
        self.open = dfcols["open"]
        self.low = dfcols["low"]
        self.high = dfcols["high"]
        self.volume = dfcols["volume"]
        self.price_ma = price_ma
        self.volume_ma = volume_ma

    def _create_features(self, data: pd.DataFrame):
        """Create the features according to price_ma and volume_ma lengths"""
        data["price_MA"] = data[self.close].rolling(window=self.price_ma).mean()
        data["volume_MA"] = data[self.volume].rolling(window=self.volume_ma).mean()
        data.dropna(inplace=True)  # remove nan
        return data

    def prime_data(self, data: pd.DataFrame):
        """Prime data for algorithm"""
        data = self._create_features(data)
        conditions = [
            (data["price_MA"] > data[self.close])
            & (data[self.volume] > data["volume_MA"]),
            (data["price_MA"] < data[self.close])
            & (data[self.volume] < data["volume_MA"]),
        ]
        choices = ["buy", "sell"]
        data["signal"] = np.select(conditions, choices, default="hold")
        return data

    def get_signal(self, data: pd.DataFrame):
        """Just get the signal"""
        if data.shape[0] < self.price_ma or data.shape[0] < self.volume_ma:
            logger.warning(
                f"Length of dataset < moving avg length required ({data.shape[0]} < {self.price_ma} or {self.volume_ma})"
            )
            return "hold"
        data = self.prime_data(data)
        signal = data.signal[-1]
        return signal

    def backtest(self, data=pd.DataFrame, loud=True):
        pass


if __name__ == "__main__":
    ape = Alpaca()
    mom = Momentum()
    data = ape.get_bars("pg")
    # signal = mom.get_signal(data)
    # logger.info(f"SIGNAL: {signal}")
    print(mom.backtest(data))