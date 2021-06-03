import logging, os

from datetime import datetime, timedelta

from channels.alpaca import Alpaca, TimeFrame
from channels.binanceus import BinanceUS
from strategies.dnn import FEDNNStrategy
from strategies.knearest import FEKNNStrategy
from random import shuffle

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
    symbol = "iht"
    ape = Alpaca()
    data = ape.get_bars(
        symbol,
        start_time=datetime.now() - timedelta(days=10),
        end_time=datetime.now(),
    )

    # create knn
    knn = FEKNNStrategy()
    knn.tune(data)
    # evaluate
    knn.evaluate(data, tt_split=0.8, securityname=symbol)
    # create dnn
    dnn = FEDNNStrategy()
    # evaluate
    dnn.evaluate(data, tt_split=0.8, securityname=symbol)