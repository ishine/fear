"""
Class designed for inheritance
"""
import logging
import os
from abc import abstractmethod
from datetime import datetime, timedelta
from threading import Thread

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import threading
from multiprocessing import Pool, Process
from time import sleep

import numpy as np
import pandas as pd
import pytz
import tensorflow as tf
from abraham3k.prophets import Abraham
from db.tables import Sentiment
from db.db import *

DTFORMAT = "%Y-%m-%dT%H:%M:%SZ"

tf.get_logger().setLevel("ERROR")
tf.autograph.set_verbosity(1)

pd.options.mode.chained_assignment = None  # default='warn'

os.environ["NUMEXPR_MAX_THREADS"] = "8"

logging.basicConfig(
    # filename,
    format="[%(name)s] %(levelname)s %(asctime)s %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

MODELPATH = "models"
CHARTPATH = "chart"


class BaseCycler:
    def __init__(self, connection, username, password, host, port, database) -> None:
        """
        Example:
        def __init__(self,
        api_key: str = open("keys/alpaca_paper_public").read().strip(),
        api_secret: str = open("keys/alpaca_paper_private").read().strip(),
        base_url: str = "https://paper-api.alpaca.markets",):

            self.alpaca = Alpaca(api_key=api_key, api_secret=api_secret, base_url=base_url)
        """

        self.abraham = Abraham(
            news_source="newsapi",
            # newsapi_key=open("keys/newsapi-public-2").read().strip(),
            bearer_token=open("keys/twitter-bearer-token").read().strip(),
        )
        self.engine = build_engine(connection, username, password, host, port, database)

    def trade(self, symbol: str, side: str, price: float, qty: int):
        """Trades if all checks pass"""
        buying_power = self.get_buying_power()

        try:
            if side == "buy":
                if buying_power >= price * qty:
                    self.submit_limit_order(
                        symbol=symbol, side=side, price=price, qty=qty
                    )
                    return True
                else:
                    logger.warning(
                        f"Not enough balance to buy ({buying_power} < {price*qty})"
                    )
            elif side == "sell":
                self.submit_limit_order(symbol=symbol, side=side, price=price, qty=qty)
                return True
            elif side == "hold":  # do nothing
                return True
        except Exception as e:
            logger.warning(
                f"Couldn't place order ({e}). Is shorting enabled on your account?"
            )
        return False

    def cycle_sentiment(
        self,
        search_term,
        period: timedelta = timedelta(minutes=30),
        news_interval=timedelta(hours=4),
        until: datetime = datetime(2022, 1, 1),
    ):
        """Cycle saving the sentiment every period
        period = how long to sleep for
        news_interval = how far back to include news data
        """
        logger.info(
            f"Start sentiment cycle for {search_term} with period={period} and news_interval={news_interval}"
        )
        sleep_sec = period.total_seconds()

        while datetime.now() <= until:
            # save
            try:
                sentiment = self.abraham.news_summary(
                    [search_term], start_time=(datetime.now() - news_interval)
                )[search_term]
                positive = sentiment[0]

                ormmap = Sentiment(
                    close_time=datetime.now().replace(
                        second=0, microsecond=0, tzinfo=pytz.timezone("US/Eastern")
                    ),
                    open_time=datetime.now() - news_interval,
                    search_term=search_term,
                    sentiment=positive,
                )
                add(self.engine, ormmap)
                logger.info(f"Got sentiment data for '{search_term}': {sentiment}")
            except Exception as e:
                logger.warning(f"Error getting sentiment data for {search_term} ({e})")
            logger.info(f"Sleeping {period} ...")
            sleep(sleep_sec)

    def cycle(
        self,
        symbol: str,
        search_term: str,
        train_period: timedelta = timedelta(minutes=30),
    ):
        """
        first train then start
        # build the model
        data = self.binanceus.get_bars(
            symbol,
            # timeframe=TimeFrame.Minute,
            start_time=datetime.now() - timedelta(days=14),
        )
        self.build(data)
        self.train(data)

        self.cycle_trades_REST(symbol)

        """
        logger.info("Starting training thread")
        training_thread = Thread(
            target=self.cycle_train,
            args=(
                symbol,
                train_period,
            ),
        )
        training_thread.start()

        logger.info("Starting sentiment thread")
        sentiment_thread = Thread(
            target=self.cycle_sentiment,
            args=(search_term,),
        )
        sentiment_thread.start()

        logger.info("Starting trading thread")
        trading_thread = Thread(
            target=self.cycle_trades,
            args=(symbol,),
        )
        trading_thread.start()

        # join
        training_thread.join()
        sentiment_thread.join()
        trading_thread.join()

    @abstractmethod
    def cycle_train(self, symbol: str, period: timedelta = timedelta(minutes=30)):
        """
        Cycles training the prediction model

        Params
        ------
        symbol : str
            stock symbol
        train_interval : int
            how many minutes between trains

        EXAMPLE:

        interval = 60 * period.total_seconds()  # convert to minutes

        # build the model
        data = self.get_data(symbol=symbol)
        self.build_models(data)

        while True:
            data = self.get_data(symbol=symbol)
            self.train_models(data)
            logger.info(f"Sleeping {period} ...")
            sleep(interval)

        """
        pass

    @abstractmethod
    def cycle_trades(self, symbol: str):
        """Cycle the trading. Can be stream or rest"""
        pass

    @abstractmethod
    def build_models(self, data):
        """
        Build model to be implemented
        """
        pass

    @abstractmethod
    def train_models(self, data):
        """
        Train model to be implemented
        """
        pass

    @abstractmethod
    def get_data(self, symbol: str, **kwargs):
        """This could be pulling and processing from the database"""
        pass

    @abstractmethod
    def get_signal(self, datà):
        """Get the signals"""
        pass

    @abstractmethod
    def submit_limit_order(self, symbol: str, side: str, price: float, qty: int = 1):
        """Submit order
        Example:
        self.alpaca.submit_limit_order(
            symbol=symbol, side=side, price=price, qty=qty
        )
        """
        pass
