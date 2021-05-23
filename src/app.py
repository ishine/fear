import logging, os

from datetime import datetime, timedelta

from channels.alpaca import Alpaca, TimeFrame
from channels.binanceus import BinanceUS
from dnn import FEDNN
from random import shuffle

# logging to file
LOGPATH = "logs"

if not os.path.exists(LOGPATH):
    os.mkdir(LOGPATH)

logging.basicConfig(
    filename=os.path.join(
        LOGPATH, "log_" + datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
    ),
    format="[%(name)s] %(levelname)s %(asctime)s %(message)s",
    level=logging.INFO,
)

logger = logging.getLogger("app")

if __name__ == "__main__":
    ape = Alpaca()
    bnc = BinanceUS()

    symbols = ["iht", "tsla", "aal", "fb", "pg", "aapl", "bdry"]
    # symbols = ["tsla"]
    # symbols = ["BTCUSD", "ETHUSD"]
    shuffle(symbols)
    logger.info(f"Testing on {symbols}")
    for symbol in symbols:
        data = ape.get_bars(
            symbol,
            timeframe=TimeFrame.Minute,
            start_time=datetime.now() - timedelta(days=14),
            end_time=datetime.now() - timedelta(days=3),
        )

        # create fednn
        fednn = FEDNN(epochs=25)
        # evaluate
        fednn.evaluate(data, tt_split=0.8, securityname=symbol)
