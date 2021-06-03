from enum import Enum
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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

logging.basicConfig(
    # filename,
    format="[%(name)s] %(levelname)s %(asctime)s %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class FEKNN(BaseTrader):
    def __init__(
        self,
        window: int = 20,
        lags: int = 6,
        n_neighbors: int = 65,
    ) -> None:
        """
        window: analysis window
        """
        super(FEKNN, self).__init__(lags=lags)
        self.n_neighbors = n_neighbors
        self.window = window

    def _create_features(self, data: pd.DataFrame):
        """Create the features"""
        data = data.copy()
        data["vol"] = data["return"].rolling(self.window).std()
        data["mom"] = np.sign(data["return"].rolling(self.window).mean())
        data["sma"] = data["close"].rolling(self.window).mean()
        data["min"] = data["close"].rolling(self.window).min()
        data["max"] = data["close"].rolling(self.window).max()

        data["MA_Fast"] = data["close"].ewm(span=13, min_periods=13).mean()
        data["MA_Slow"] = data["close"].ewm(span=24, min_periods=24).mean()
        data["MACD"] = data["MA_Fast"] - data["MA_Slow"]
        data["MACD_signal"] = data["MACD"].ewm(span=9, min_periods=9).mean()

        data = self._append_data(data, ta.RSI(data))
        # self.features.extend(["vol", "mom", "sma", "min", "max", "14 period RSI"])
        self._add_columns(
            "close",
            "MA_Fast",
            "MA_Slow",
            "MACD",
            "MACD_signal",
            "14 period RSI",
            "vol",
            "mom",
            "sma",
            "min",
            "max",
        )
        return data

    def build(self, data: pd.DataFrame, summary: bool = False):
        """Compile model"""
        data = data.copy()
        data = self.prime_data(data)
        logger.info(f"Building model using features {self.cols}")

        self.model_unfitted = KNeighborsClassifier(n_neighbors=self.n_neighbors)

    def train(self, data: pd.DataFrame):
        """Train the model"""
        data = data.copy()
        data = self.prime_data(data)
        logger.info(f"Training model on {data.shape[0]} records")
        self.model = self.model_unfitted.fit(data[self.cols], data["direction"])
        validation = self.model.predict(data[self.cols])
        accuracy_in_sample = accuracy_score(data["direction"], validation)
        logger.info(f"In-sample accuracy={accuracy_in_sample:.4f}")

    def predict(self, data: pd.DataFrame):
        """Predict
        Strict hold = introduce hold signals (0) as well, not just 1 and -1
        """
        data = data.copy()
        data = self.prime_data(data)

        data["prediction"] = self.model.predict(data[self.cols])
        return data

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
        # prime data happens inside each function
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
        # make model
        self.build(train)

        # train model
        self.train(train)

        # predict
        predictions = self.predict(test)[["close", "return", "prediction"]]

        # calcluate returns
        predictions["strategy"] = predictions["prediction"] * predictions["return"]

        # count trades
        num_trades = (predictions["prediction"] != 0).sum()

        logger.info(f"Trades made: {num_trades}")

        # define returns
        predictions["return"] = predictions["return"].cumsum().apply(np.exp)
        predictions["strategy"] = predictions["strategy"].cumsum().apply(np.exp)

        predictions["buys"] = (
            np.where(predictions["prediction"] == -1, 1, NaN) * predictions["close"]
        )
        predictions["sells"] = (
            np.where(predictions["prediction"] == 1, 1, NaN) * predictions["close"]
        )

        returns = predictions[["return", "strategy"]]  # .cumsum().apply(np.exp)

        logger.info(f"Returns [{securityname}]:\n{returns.tail(1)}")
        # write to csv and stuff
        if securityname:
            self.save_plot(predictions, securityname)
        return returns

    def tune(self, data: pd.DataFrame, k_range: range = range(1, 70)):
        """Search a k range and find the best param"""
        values = pd.DataFrame(columns=["k", "strategy"])
        for k in k_range:
            self.n_neighbors = k
            returns = self.evaluate(data, tt_split=0.7).tail(1)[["strategy"]]
            returns["k"] = k
            values = values.append(returns.iloc[[-1]])
        best = values.iloc[values["strategy"].values.argmax()]["k"]
        self.n_neighbors = best
        return best


if __name__ == "__main__":
    symbol = "iht"
    ape = Alpaca()
    data = ape.get_bars(
        symbol,
        start_time=datetime.now() - timedelta(days=10),
        end_time=datetime.now(),
    )

    # create feknn
    feknn = FEKNN()
    # evaluate
    feknn.tune(data)
    feknn.evaluate(data, tt_split=0.7, securityname=symbol)
