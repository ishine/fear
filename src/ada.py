from sys import exc_info
import numpy as np, pandas as pd
import logging
from channels.alpaca import Alpaca, TimeFrame
from datetime import datetime, timedelta
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

from finta import TA as ta


logging.basicConfig(
    # filename,
    format="[%(name)s] %(levelname)s %(asctime)s %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class ADA:
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
        self.window = window
        self.lags = lags
        self.features = features
        self.cols = []  # cols different than features here
        self.estimators = estimators
        self.random_state = random_state
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample

    def _create_returns(self, data: pd.DataFrame):
        """Create the log returns"""
        data = data.copy()
        try:
            data["return"] = np.log(data["close"] / data["close"].shift(1))
        except Exception as e:
            logger.warning(f"Unknown error occured while generating returns ({e})")
        return data

    def _create_direction(self, data: pd.DataFrame):
        """Create direction from returns"""
        data = data.copy()
        try:
            data["direction"] = np.where(data["return"] > 0, 1, -1)
        except Exception as e:
            logger.warning(f"Unknown error occured while generating direction ({e})")
        return data

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

    def _append_data(
        self, maindf: pd.DataFrame, dataarray: list, namesarray: list = None
    ):
        if namesarray == None:
            return maindf.join(pd.DataFrame(dataarray), how="outer")
        return maindf.join(pd.DataFrame(dataarray, columns=namesarray), how="outer")

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

    def prime_data(self, data: pd.DataFrame, prune: bool = False):
        """Prime the data for the network"""
        data = data.copy()
        if prune:
            data = data[["close"]]

        data = self._create_returns(data)
        data = self._create_direction(data)
        data = self._create_features(data)
        data = self._create_lags(data)
        self.cols = list(set(self.cols))  # remove duplicates
        # logger.info("Primed data")
        return data

    def fit_scaler(self, data: pd.DataFrame):
        """Fit the min max scaler - MUST BE CALLED BEFORE SCALING"""
        self.mu, self.std = data.mean(), data.std()

    def normalize(self, data: pd.DataFrame):
        """Normalize the data and return"""
        data = data.copy()
        normalized = (data - self.mu) / self.std
        return normalized

    def build(self, data: pd.DataFrame, summary: bool = False):
        """Compile model"""
        data = data.copy()
        data = self.prime_data(data)
        logger.info(f"Building model using features {self.cols}")
        self.fit_scaler(data)
        self.dtc = DecisionTreeClassifier(
            random_state=self.random_state,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
        )
        self.model = AdaBoostClassifier(
            base_estimator=self.dtc,
            n_estimators=self.estimators,
            random_state=self.random_state,
        )

    def train(self, data: pd.DataFrame):
        """Train the model"""
        data = data.copy()
        data = self.prime_data(data)
        logger.info(f"Training model on {data.shape[0]} records")
        data_normalized = self.normalize(data)
        self.model.fit(data[self.cols], data["direction"])
        accuracy = accuracy_score(
            data["direction"], self.model.predict(data[self.cols])
        )
        logger.info(f"Accuracy score (in sample) = {accuracy}")

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
        returns = predictions[["return", "strategy"]]  # .cumsum().apply(np.exp)

        logger.info(f"Returns [{securityname}]:\n{returns.tail(1)}")


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
