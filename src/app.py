import logging
import os
from datetime import datetime, timedelta
from random import shuffle

from channels.alpaca import Alpaca, TimeFrame
from channels.binanceus import BinanceUS
from strategies_support.predictors import FEDNNPredictor, FEKNNPredictor

# logging

LOGPATH = "logs"

if not os.path.exists(LOGPATH):
    os.mkdir(LOGPATH)

filename = os.path.join(
    LOGPATH, "log_" + datetime.now().strftime("%m-%d-%Y_%H-%M-%S") + ".log"
)

logging.basicConfig(
    # filename,
    format="[%(name)s] %(levelname)s %(asctime)s %(message)s",
    level=logging.INFO,
)

logger = logging.getLogger("app")


if __name__ == "__main__":

    symbols = ["tsla", "iht", "spy"]
    ape = Alpaca()
    for symbol in symbols:
        data = ape.get_bars(
            symbol,
            start_time=datetime.now() - timedelta(days=14),
            end_time=datetime.now(),
            resample=1,
        )
        # create knn
        knn = FEKNNPredictor()
        # knn.tune(data)
        # evaluate
        knn.evaluate(data, tt_split=0.8, securityname=symbol)
        # create predictors
        predictors = FEDNNPredictor()
        # evaluate
        predictors.evaluate(data, tt_split=0.8, securityname=symbol)
