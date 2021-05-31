from channels.screener import Screener
from datetime import datetime
import numpy as np
import pandas as pd
from pylab import mpl, plt
from channels.alpaca import Alpaca, TimeFrame

plt.style.use("seaborn")
mpl.rcParams["font.family"] = "serif"


class BacktestBase(object):
    """Base class for event-based backtesting of trading strategies.

    Attributes
    ==========
    symbol: str
        TR RIC (financial instrument) to be used
    start: str
        start date for data selection
    end: str
        end date for data selection
    amount: float
        amount to be invested either once or per trade
    ftc: float
        fixed transaction costs per trade (buy or sell)
    ptc: float
        proportional transaction costs per trade (buy or sell)

    Methods
    =======
    get_data:
        retrieves and prepares the base data set
    plot_data:
        plots the closing price for the symbol
    get_date_price:
        returns the date and price for the given bar
    print_balance:
        prints out the current (cash) balance
    print_net_wealth:
        prints auf the current net wealth
    place_buy_order:
        places a buy order
    place_sell_order:
        places a sell order
    close_out:
        closes out a long or short position
    """

    def __init__(self, symbol, amount, ftc=0.0, ptc=0.0, verbose=True):
        self.symbol = symbol
        self.initial_amount = amount
        self.amount = amount
        self.ftc = ftc
        self.ptc = ptc
        self.units = 0
        self.position = 0
        self.trades = 0
        self.verbose = verbose
        self.alpaca = Alpaca()
        self.get_data()

    def get_data(self):
        """Retrieves and prepares the data."""
        raw = self.alpaca.get_bars(
            self.symbol,
            timeframe=TimeFrame.Hour,
            start_time=datetime(2021, 1, 3, 0, 0, 0),
            end_time=datetime(2021, 5, 29, 23, 59, 0),
        )
        raw.rename(columns={"close": "price"}, inplace=True)
        raw["return"] = np.log(raw["price"] / raw["price"].shift(1))
        self.data = raw.dropna()

    def plot_data(self, cols=None):
        """Plots the closing prices for symbol."""
        if cols is None:
            cols = ["price"]
        self.data["price"].plot(figsize=(10, 6), title=self.symbol)

    def get_date_price(self, bar):
        """Return date and price for bar."""
        date = str(self.data.index[bar])[:10]
        price = self.data.price.iloc[bar]
        return date, price

    def print_balance(self, bar):
        """Print out current cash balance info."""
        date, price = self.get_date_price(bar)
        print(f"{date} | current balance {self.amount:.2f}")

    def print_net_wealth(self, bar):
        """Print out current cash balance info."""
        date, price = self.get_date_price(bar)
        net_wealth = self.units * price + self.amount
        print(f"{date} | current net wealth {net_wealth:.2f}")

    def place_buy_order(self, bar, units=None, amount=None):
        """Place a buy order."""
        date, price = self.get_date_price(bar)
        if units is None:
            units = int(amount / price)
        self.amount -= (units * price) * (1 + self.ptc) + self.ftc
        self.units += units
        self.trades += 1
        if self.verbose:
            print(f"{date} | selling {units} units at {price:.2f}")
            self.print_balance(bar)
            self.print_net_wealth(bar)

    def place_sell_order(self, bar, units=None, amount=None):
        """Place a sell order."""
        date, price = self.get_date_price(bar)
        if units is None:
            units = int(amount / price)
        self.amount += (units * price) * (1 - self.ptc) - self.ftc
        self.units -= units
        self.trades += 1
        if self.verbose:
            print(f"{date} | selling {units} units at {price:.2f}")
            self.print_balance(bar)
            self.print_net_wealth(bar)

    def close_out(self, bar):
        """Closing out a long or short position."""
        date, price = self.get_date_price(bar)
        self.amount += self.units * price
        self.units = 0
        self.trades += 1
        if self.verbose:
            print(f"{date} | inventory {self.units} units at {price:.2f}")
            print("=" * 55)
        print(
            "Final balance   [$] {:.2f}".format(
                self.amount,
            )
        )
        perf = (self.amount - self.initial_amount) / self.initial_amount * 100
        print(
            "Net Performance [%] {:.2f} (vs {:.2f} for buy & hold)".format(
                perf,
                (
                    (
                        (self.data["price"][-1] - self.data["price"][0])
                        / self.data["price"][0]
                    )
                    * 100
                ),
            )
        )
        print("Trades Executed [#] {:.2f}".format(self.trades))
        print("=" * 55)


class BacktestLongOnly(BacktestBase):
    def run_sma_strategy(self, SMA1, SMA2):
        """Backtesting a SMA-based strategy.

        Parameters
        ==========
        SMA1, SMA2: int
            shorter and longer term simple moving average (in days)
        """
        msg = f"\n\nRunning SMA strategy | SMA1={SMA1} & SMA2={SMA2}"
        msg += f"\nfixed costs {self.ftc} | "
        msg += f"proportional costs {self.ptc}"
        print(msg)
        print("=" * 55)
        self.position = 0  # initial neutral position
        self.trades = 0  # no trades yet
        self.amount = self.initial_amount  # reset initial capital
        self.data["SMA1"] = self.data["price"].rolling(SMA1).mean()
        self.data["SMA2"] = self.data["price"].rolling(SMA2).mean()

        for bar in range(SMA2, len(self.data)):
            if self.position == 0:
                if self.data["SMA1"].iloc[bar] > self.data["SMA2"].iloc[bar]:
                    self.place_buy_order(bar, amount=self.amount)
                    self.position = 1  # long position
            elif self.position == 1:
                if self.data["SMA1"].iloc[bar] < self.data["SMA2"].iloc[bar]:
                    self.place_sell_order(bar, units=self.units)
                    self.position = 0  # market neutral
        self.close_out(bar)

    def run_momentum_strategy(self, momentum):
        """Backtesting a momentum-based strategy.

        Parameters
        ==========
        momentum: int
            number of days for mean return calculation
        """
        msg = f"\n\nRunning momentum strategy | {momentum} days"
        msg += f"\nfixed costs {self.ftc} | "
        msg += f"proportional costs {self.ptc}"
        print(msg)
        print("=" * 55)
        self.position = 0  # initial neutral position
        self.trades = 0  # no trades yet
        self.amount = self.initial_amount  # reset initial capital
        self.data["momentum"] = self.data["return"].rolling(momentum).mean()
        for bar in range(momentum, len(self.data)):
            if self.position == 0:
                if self.data["momentum"].iloc[bar] > 0:
                    self.place_buy_order(bar, amount=self.amount)
                    self.position = 1  # long position
            elif self.position == 1:
                if self.data["momentum"].iloc[bar] < 0:
                    self.place_sell_order(bar, units=self.units)
                    self.position = 0  # market neutral
        self.close_out(bar)


if __name__ == "__main__":

    def run_strategies():
        lobt.run_sma_strategy(42, 252)
        lobt.run_momentum_strategy(60)

    screener = Screener()
    symbols = screener.get_losers()["symbol"]
    for symbol in symbols:
        lobt = BacktestLongOnly(symbol, 10000, verbose=False)
        lobt.run_momentum_strategy(60)
