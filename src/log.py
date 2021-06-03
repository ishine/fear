from numpy.core.numeric import NaN
from base import BaseStrategy
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


logging.basicConfig(
    # filename,
    format="[%(name)s] %(levelname)s %(asctime)s %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class ADA(BaseStrategy):
    def __init__(
        self,
        window: int = 20,
        lags: int = 6,
        features: list = ["return"],
        estimators: int = 15,
        random_state: int = 100,
        max_depth: int = 2,
        min_samples_leaf: int = 1,
        subsample: float = 0.33,
    ) -> None:
        """
        window: analysis window
        """
        super(ADA, self).__init__(lags=lags)
        self.window = window
        self.features = features
        self.cols = []  # cols different than features here
        self.estimators = estimators
        self.random_state = random_state
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample

    def _create_lags(self, data: pd.DataFrame):
        """Create the lags"""
        data = data.copy()
        try:
            for f in self.features:
                for lag in range(1, self.lags + 1):
                    col = f"{f}_lag_{lag}"
                    data[col] = data["return"].shift(lag)
                    self.cols.append(col)
                data.dropna(inplace=True)
        except Exception as e:
            logger.warning(f"Unknown error occured while generating lags ({e})")
        return data

    def _create_features(self, data: pd.DataFrame):
        """Create the features"""
        data = data.copy()
        data["vol"] = data["return"].rolling(self.window).std
        data["mom"] = np.sign(data["return"].rolling(self.window).mean())
        data["sma"] = data["close"].rolling(self.window).mean()
        data["min"] = data["close"].rolling(self.window).min()
        data["max"] = data["close"].rolling(self.window).max()
        data = self._append_data(data, ta.RSI(data))
        self.features.extend(["vol", "mom", "sma", "min", "max", "14 period RSI"])
        return data

    def build(self, data: pd.DataFrame, summary: bool = False):
        """Compile model"""
        data = data.copy()
        data = self.prime_data(data)
        logger.info(f"Building model using features {self.cols}")
        self.fit_scaler(data)

    def train(self, data: pd.DataFrame):
        """Train the model"""
        data = data.copy()
        data = self.prime_data(data)
        logger.info(f"Training model on {data.shape[0]} records")
        data_normalized = self.normalize(data)

    def predict(self, data: pd.DataFrame):
        """Predict
        Strict hold = introduce hold signals (0) as well, not just 1 and -1
        """
        data = data.copy()
        data = self.prime_data(data)
        data_normalized = self.normalize(data)
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
        data = data.copy()
        """Vectorize evaluate - split data into train and test, build, and evaluate"""
        # prime data happens inside each function

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


if __name__ == "__main__":
    symbol = "iht"
    ape = Alpaca()
    data = ape.get_bars(
        symbol,
        start_time=datetime.now() - timedelta(days=10),
        end_time=datetime.now(),
    )

    # create fednn
    ada = ADA()
    # evaluate
    ada.evaluate(data, tt_split=0.8, securityname=symbol)
