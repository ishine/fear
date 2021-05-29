import logging, os

from datetime import datetime, timedelta

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

ape = Alpaca()
bnc = BinanceUS()


def test_w_stocks(symbols, strict_hold=False):
    shuffle(symbols)
    for symbol in symbols:
        try:
            data = ape.get_bars(
                symbol,
                timeframe=TimeFrame.Minute,
                start_time=datetime.now() - timedelta(days=10),
                end_time=datetime.now(),
            )

            # create fednn
            fednn = FEDNN(epochs=25)
            # evaluate
            fednn.evaluate(
                data, tt_split=0.89, securityname=symbol, strict_hold=strict_hold
            )
        except Exception as e:
            logging.warning(f"Couldn't do {symbol} ({e})")


def test_w_crypto(symbols, strict_hold=False):
    shuffle(symbols)
    for symbol in symbols:
        try:
            data = bnc.get_bars(
                symbol,
                start_time=datetime.now() - timedelta(days=10),
                end_time=datetime.now(),
            )

            # create fednn
            fednn = FEDNN(epochs=25)
            # evaluate
            fednn.evaluate(
                data, tt_split=0.8, securityname=symbol, strict_hold=strict_hold
            )
        except Exception as e:
            logging.warning(f"Couldn't do {symbol} ({e})")


if __name__ == "__main__":
    cryptosymbols = [
        "BTCUSD",
        "ETHUSD",
        "ADAUSD",
    ]
    stocksymbols = [
        "iht",
        "tsla",
        "aal",
        "fb",
        "aapl",
        "bdry",
        "spce",
        "ocft",
        "gme",
        "amc",
        "snap",
        "tal",
        "tuya",
        "cog",
    ]
    test_w_stocks(stocksymbols, strict_hold=False)
    test_w_crypto(cryptosymbols, strict_hold=True)
