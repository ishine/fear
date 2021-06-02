from numpy.core.numeric import NaN
from mlbase import BaseTrader
from sys import exc_info
import numpy as np, pandas as pd
import logging, warnings
from channels.alpaca import Alpaca, TimeFrame
from datetime import datetime, timedelta
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import linear_model
from finta import TA as ta
from statsmodels.tsa.api import VAR as StatsVAR
from statsmodels.tsa.base.tsa_model import ValueWarning

warnings.simplefilter(action="ignore", category=ValueWarning)

logging.basicConfig(
    # filename,
    format="[%(name)s] %(levelname)s %(asctime)s %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class VAR(BaseTrader):
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
        super(VAR, self).__init__(lags=lags)
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
        data["vol"] = data["return"].rolling(self.window).std()
        data["mom"] = np.sign(data["return"].rolling(self.window).mean())
        data["sma"] = data["close"].rolling(self.window).mean()
        data["min"] = data["close"].rolling(self.window).min()
        data["max"] = data["close"].rolling(self.window).max()
        data = self._append_data(data, ta.RSI(data))
        self.features.extend(["vol", "mom", "sma", "min", "max", "14 period RSI"])
        return data

    def predict(self, data: pd.DataFrame, start_at: int = 1):
        """Predict
        start at = where to start
        """
        data = data.copy()
        data = self.prime_data(data)
        self.fit_scaler(data)
        data_normalized = self.normalize(data)
        col_loc = data.columns.get_loc("direction")
        pred = []
        for pos in range(start_at, data.shape[0]):
            sliced = data.iloc[:pos]
            unfit_model = StatsVAR(sliced)
            self.model = unfit_model.fit()
            next_vals = self.model.forecast(self.model.endog, steps=1)[0]
            next_direction = round(next_vals[col_loc])
            pred.append(next_direction)  # get just direction
        truncated = data.iloc[start_at:]
        truncated["prediction"] = pred
        print(pred)
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

        self.predict(data, start_at=int(data.shape[0] * tt_split))


if __name__ == "__main__":
    symbol = "iht"
    ape = Alpaca()
    data = ape.get_bars(
        symbol,
        start_time=datetime.now() - timedelta(days=10),
        end_time=datetime.now(),
    )

    # create fednn
    var = VAR()
    # evaluate
    var.evaluate(data, tt_split=0.8, securityname=symbol)
