from numpy.core.numeric import NaN
from strategies.base import BaseStrategy
import numpy as np, pandas as pd
import logging
from datetime import datetime, timedelta
from sklearn.metrics import accuracy_score
from finta import TA as ta
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

logging.basicConfig(
    # filename,
    format="[%(name)s] %(levelname)s %(asctime)s %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class MOMStrategy(BaseStrategy):
    def __init__(self, lags: int = 6, momentum: int = 30) -> None:
        """
        window: analysis window
        """
        super(MOMStrategy, self).__init__(lags=lags, momentum=momentum)

    def prime_data(self, data: pd.DataFrame, prune: bool = False):
        """Reimplement for the momentum"""
        # data = super().prime_data(data, prune=prune)
        data = self._create_returns(data)
        data = self._create_momentum(data)
        return data