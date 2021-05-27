from channels.binanceus import BinanceUS
from channels.alpaca import Alpaca, TimeFrame
from dnn import FEDNN
import threading
import pandas as pd, numpy as np
from time import sleep
from datetime import date, datetime, timedelta
import logging, math, os
from threading import Thread

logger = logging.getLogger(__name__)

LOGPATH = "logs"

if not os.path.exists(LOGPATH):
    os.mkdir(LOGPATH)

TRACKPATH = os.path.join(
    LOGPATH, "track_" + datetime.now().strftime("%m-%d-%Y_%H-%M-%S" + ".csv")
)


class BinanceUSCycler:
    """We will NOT trade crypto, this is just for testing algorithm"""

    def __init__(
        self,
        api_key: str = open("keys/binanceus-public").read().strip(),
        api_secret: str = open("keys/binanceus-private").read().strip(),
    ) -> None:
        self.fednn = FEDNN()
        self.binanceus = BinanceUS(api_key, api_secret)
        self.track = pd.DataFrame(
            columns=["price", "return", "signal", "prediction", "strategy"]
        )

    def _add_track(self, price: float, prediction: int, signal: str):
        self.track = self.track.append(
            pd.DataFrame(
                {"price": [price], "prediction": [prediction], "signal": signal}
            )
        )
        self.track["return"] = np.log(
            self.track["price"] / self.track["price"].shift(1)
        )
        self.track["strategy"] = (
            (self.track["return"] * self.track["prediction"]).cumsum().apply(np.exp)
        )
        self.track.to_csv(TRACKPATH)

    def trade(self, ticker: str, side: str, price: float, qty: int):
        """Trades if all checks pass"""
        return False

    def cycle_trades(self, ticker: str):
        """
        Cycles between predicting and trading

        Params
        ------
        ticker : str
            stock ticker
        spend_amt : int = 1000
            max amount of money to spend
        """

        while True:
            # get the data
            data = self.binanceus.get_bars(
                ticker,
                start_time=datetime.now() - timedelta(hours=12),
                end_time=datetime.now(),
            )
            close = data.close
            price = close[-1]
            qty = 1

            prediction = self.fednn.get_signal(data)

            if prediction == -1:
                signal = "buy"
            elif prediction == 1:
                signal = "sell"
            else:
                signal = "hold"
            self._add_track(price, prediction, signal)
            logger.info(f"{signal} {qty} @ {price:.2f}")
            # act
            # self.trade(ticker, signal, price, qty)
            # sleep til next min start
            logger.info(f"Sleeping {60 - datetime.now().second} s ...")
            sleep(60 - datetime.now().second)

    def cycle_train(self, ticker: str, train_interval: int = 30):
        """
        Cycles training the prediction model

        Params
        ------
        ticker : str
            stock ticker
        train_interval : int
            how many minutes between trains
        """
        # model must have been built first
        interval = 60 * train_interval  # convert to seconds

        # build the model
        data = self.binanceus.get_bars(
            ticker,
            # timeframe=TimeFrame.Minute,
            start_time=datetime.now() - timedelta(days=14),
        )
        self.fednn.build(data)

        while True:
            self.fednn.train(data)
            logger.info(f"Sleeping {interval} mins ...")
            sleep(interval)
            data = self.binanceus.get_bars(
                ticker,
                # timeframe=TimeFrame.Minute,
                start_time=datetime.now() - timedelta(days=14),
            )

    def cycle(self, ticker: str, train_interval: int = 30):
        """
        first train then start
        # build the model
        data = self.binanceus.get_bars(
            ticker,
            # timeframe=TimeFrame.Minute,
            start_time=datetime.now() - timedelta(days=14),
        )
        self.fednn.build(data)
        self.fednn.train(data)

        self.cycle_trades(ticker)

        """
        logger.info("Starting training thread")
        training_thread = Thread(
            target=self.cycle_train,
            args=(
                ticker,
                train_interval,
            ),
        )
        training_thread.start()
        sleep(60)
        logger.info("Starting trading thread")
        trading_thread = Thread(
            target=self.cycle_trades,
            args=(ticker,),
        )
        trading_thread.start()
        # join
        training_thread.join()
        trading_thread.join()


class AlpacaCycler:
    def __init__(
        self,
        api_key: str = open("keys/alpaca_paper_public").read().strip(),
        api_secret: str = open("keys/alpaca_paper_private").read().strip(),
        base_url: str = "https://paper-api.alpaca.markets",
    ) -> None:
        self.fednn = FEDNN()
        self.alpaca = Alpaca()
        self.track = pd.DataFrame(columns=["price", "prediction"])

    def _add_track(self, price: float, prediction: int):
        self.track = self.track.append(
            pd.DataFrame({"price": [price], "prediction": [prediction]})
        )

    def trade(self, ticker: str, side: str, price: float, qty: int):
        """Trades if all checks pass"""
        buying_power = self.alpaca.get_buying_power()
        num_shares = self.alpaca.get_shares(ticker)
        open_trades = self.alpaca.any_open_orders()

        try:
            if side == "buy":
                if buying_power >= price * qty:
                    self.alpaca.submit_limit_order(
                        ticker=ticker, side=side, price=price, qty=qty
                    )
                    return True
                else:
                    logger.warning(
                        f"Not enough balance to buy ({buying_power} < {price*qty})"
                    )
            elif side == "sell":
                self.alpaca.submit_limit_order(
                    ticker=ticker, side=side, price=price, qty=qty
                )
                return True
            elif side == "hold":  # do nothing
                return True
        except Exception as e:
            logger.warning(
                f"Couldn't place order ({e}). Is shorting enabled on your account?"
            )
        return False

    def cycle_trades(self, ticker: str):
        """
        Cycles between predicting and trading

        Params
        ------
        ticker : str
            stock ticker
        spend_amt : int = 1000
            max amount of money to spend
        """

        while True:
            # get the data
            data = self.alpaca.get_bars(
                ticker,
                timeframe=TimeFrame.Minute,
                start_time=datetime.now() - timedelta(hours=12),
                end_time=datetime.now(),
            )
            close = data.close
            price = close[-1]
            qty = 1

            prediction = self.fednn.get_signal(data)

            if prediction == 1:
                signal = "buy"
            elif prediction == -1:
                signal = "sell"
            else:
                signal = "hold"

            self._add_track(price, prediction)
            logger.info(f"{signal} {qty} @ {price:.2f}")
            # act
            # self.trade(ticker, signal, price, qty)
            # sleep til next min start
            logger.info(f"Sleeping {60 - datetime.now().second} s ...")
            sleep(60 - datetime.now().second)

    def cycle_train(self, ticker: str, train_interval: int = 30):
        """
        Cycles training the prediction model

        Params
        ------
        ticker : str
            stock ticker
        train_interval : int
            how many minutes between trains
        """
        # model must have been built first
        interval = 60 * train_interval  # convert to seconds

        # build the model
        data = self.alpaca.get_bars(
            ticker,
            timeframe=TimeFrame.Minute,
            start_time=datetime.now() - timedelta(days=14),
        )
        self.fednn.build(data)

        while True:
            self.fednn.train(data)
            logger.info(f"Sleeping {interval} mins ...")
            sleep(interval)
            data = self.alpaca.get_bars(
                ticker,
                timeframe=TimeFrame.Minute,
                start_time=datetime.now() - timedelta(days=14),
            )

    def cycle(self, ticker: str, train_interval: int = 30):
        """
        first train then start
        # build the model
        data = self.binanceus.get_bars(
            ticker,
            # timeframe=TimeFrame.Minute,
            start_time=datetime.now() - timedelta(days=14),
        )
        self.fednn.build(data)
        self.fednn.train(data)

        self.cycle_trades(ticker)

        """
        logger.info("Starting training thread")
        training_thread = Thread(
            target=self.cycle_train,
            args=(
                ticker,
                train_interval,
            ),
        )
        training_thread.start()
        sleep(60)
        logger.info("Starting trading thread")
        trading_thread = Thread(
            target=self.cycle_trades,
            args=(ticker,),
        )
        trading_thread.start()
        # join
        training_thread.join()
        trading_thread.join()