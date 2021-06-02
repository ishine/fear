from numpy.core.numeric import NaN
from mlbase import BaseTrader
from sys import exc_info
import numpy as np, pandas as pd
import logging
from channels.alpaca import Alpaca, TimeFrame
from datetime import datetime, timedelta
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import linear_model
from finta import TA as ta
from statsmodels.tsa.arima.model import ARIMA as StatsARIMA


logging.basicConfig(
    # filename,
    format="[%(name)s] %(levelname)s %(asctime)s %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class ARIMA(BaseTrader):
    def __init__(self, p=4, d=1, q=1):
        super(ARIMA, self).__init__()
        self.p = p
        self.q = q
        self.d = d

    def predict(self, data: pd.DataFrame, column: str = "direction", start_at: int = 1):
        """Predict"""
        data = data.copy()
        data = self.prime_data(data)
        model = StatsARIMA(
            data[column].values, order=(self.p, self.d, self.q)
        )  # create model
        fitted = model.fit()  # fit the model
        output = fitted.forecast()  # forecast ahead

        return output

    def get_signal(self, data: pd.DataFrame):
        """
        Get a signal from an already trained model
        """
        truncated = data.copy()
        predset = self.predict(truncated)

        prediction = predset.iloc[-1].prediction
        return prediction

    def evaluate(
        self,
        data: pd.DataFrame,
        tt_split: int = 0.8,
        securityname: str = None,
    ):
        """Vectorize evaluate - split data into train and test, build, and evaluate"""
        data = data.copy()

        # split
        train, test = (
            data[: int(data.shape[0] * tt_split)],
            data[int(data.shape[0] * tt_split) :],
        )

        # train test sizes
        logger.info(
            f"Train: {train.index[0]} - {train.index[-1]} ({train.shape[0]}, {len(self.cols)})"
        )
        logger.info(
            f"Test: {test.index[0]} - {test.index[-1]} ({test.shape[0]}, {len(self.cols)})"
        )


if __name__ == "__main__":
    symbol = "iht"
    ape = Alpaca()
    data = ape.get_bars(
        symbol,
        start_time=datetime.now() - timedelta(days=10),
        end_time=datetime.now(),
    )

    # create fednn
    arima = ARIMA()
    # evaluate
    arima.evaluate(data, tt_split=0.8, securityname=symbol)
