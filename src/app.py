import logging, os

from datetime import datetime, timedelta

from channels.alpaca import Alpaca, TimeFrame
from channels.binanceus import BinanceUS
from dnn import FEDNN
from random import shuffle

logging.basicConfig(
    format="[app] %(levelname)s %(asctime)s %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)
if __name__ == "__main__":
    ape = Alpaca()
    bnc = BinanceUS()

    symbols = ["iht", "tsla", "aal", "fb", "pg", "aapl", "bdry"]
    symbols = ["tsla"]
    symbols = ["BTCUSD", "ETHUSD"]
    shuffle(symbols)
    logger.info(f"Testing on {symbols}")
    for symbol in symbols:
        data = bnc.get_bars(
            symbol,
            # timeframe=TimeFrame.Minute,
            start_time=datetime.now() - timedelta(days=10),
        )

        # create fednn
        fednn = FEDNN(epochs=25)

        # evaluate
        fednn.evaluate(data, tt_split=0.96)
