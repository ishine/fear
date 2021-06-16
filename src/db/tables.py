from math import floor
from re import search
from sqlalchemy import Column, String, Integer, Table, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql.elements import and_
from sqlalchemy.sql.sqltypes import BIGINT, VARCHAR, DateTime, Float, ARRAY

from datetime import *

Base = declarative_base()


class Quote(Base):
    __tablename__ = "quotes"

    id = Column(Integer, autoincrement=True, primary_key=True)

    symbol = Column(VARCHAR(10))
    ask_exchange = Column(VARCHAR(1))
    ask_price = Column(Float)
    ask_size = Column(Float)
    bid_exchange = Column(VARCHAR(1))
    bid_price = Column(Float)
    bid_size = Column(Float)
    datetime = Column(DateTime)
    tape = Column(VARCHAR(1))

    def __repr__(self):
        tostr = f"<QUOTE @ symbol={self.symbol}, ask={self.ask_price}, bid={self.bid_price}, DateTime={self.datetime}>"
        return tostr


class Trade(Base):
    __tablename__ = "trades"

    id = Column(Integer, autoincrement=True, primary_key=True)

    symbol = Column(VARCHAR(10))
    exchange = Column(VARCHAR(1))
    price = Column(Float)
    size = Column(Integer)
    tape = Column(VARCHAR(1))
    datetime = Column(DateTime)  # maybe?

    def __repr__(self):
        tostr = f"<TRADE @ symbol={self.symbol}, price={self.price}, size={self.size}, DateTime={self.datetime}>"
        return tostr


class Bar(Base):
    __tablename__ = "bars"

    id = Column(Integer, autoincrement=True, primary_key=True)

    symbol = Column(VARCHAR(10))
    opn = Column(Float)
    close = Column(Float)
    high = Column(Float)
    low = Column(Float)
    volume = Column(Integer)  # use quote volume
    open_time = Column(DateTime)
    num_trades = Column(Integer)
    close_time = Column(DateTime)
    datetime = Column(DateTime)  # maybe?

    def __repr__(self):
        tostr = f"<BAR @ symbol={self.symbol}, opn={self.opn}, close={self.close}, low={self.low}, high={self.high}, DateTime={self.datetime}>"
        return tostr


class Sentiment(Base):
    __tablename__ = "sentiment"

    id = Column(Integer, autoincrement=True, primary_key=True)
    open_time = Column(DateTime)
    close_time = Column(DateTime)
    search_term = Column(VARCHAR(50))
    sentiment = Column(Float)

    def __repr__(self):
        tostr = f"<SENT @ search={self.search_term}, sentiment={self.sentiment}, open_time={self.open_time}, close_time={self.close_time}>"
        return tostr