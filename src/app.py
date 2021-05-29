import logging, os

from datetime import datetime, timedelta
from cycle import BinanceUSCycler

from channels.alpaca import Alpaca, TimeFrame
from channels.binanceus import BinanceUS
from dnn import FEDNN
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

cycler = BinanceUSCycler()


def test_bcycler(symbol, strict_hold=False):
    logger.info(f"Testing on {symbol}")

    cycler.cycle(symbol, strict_hold=strict_hold)


if __name__ == "__main__":
    test_bcycler("BTCUSD", strict_hold=False)
