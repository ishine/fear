# monte-carlo simulations
from typing_extensions import runtime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from channels.alpaca import Alpaca, TimeFrame
from datetime import *
import logging, os, warnings

os.environ["NUMEXPR_MAX_THREADS"] = "8"
warnings.filterwarnings(action="ignore", category=FutureWarning)
logging.basicConfig(
    format="[%(name)s] %(levelname)s %(asctime)s %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


class MonteCarlo:
    def __init__(self) -> None:
        self.alpaca = Alpaca()  # only for grabbing the data for spy capm

    def log_returns(self, data: pd.Series):
        return np.log(1 + data.pct_change())

    def simple_returns(self, data: pd.Series):
        return (data / data.shift(1)) - 1

    def _market_data_combination(
        self,
        data: pd.Series,
        mark_ticker: str = "spy",
        start_time: timedelta = datetime.now() - timedelta(days=365),
    ):
        """Combine market data with a reference against the index
        Eg combine tsla with spy into a single dataframe
        """
        data = pd.DataFrame(data)

        market_data = self.alpaca.import_stock_data([mark_ticker], start_time)
        market_rets = self.log_returns(market_data).dropna()
        ann_return = np.exp(market_rets.mean() * 252).values - 1
        data = data.merge(market_data, left_index=True, right_index=True)
        return data, ann_return

    def beta_sharpe(
        self,
        data: pd.Series,
        mark_ticker: str = "spy",
        start_time: timedelta = datetime.now() - timedelta(days=365),
        riskfree=0.025,
    ):

        """Calculate beta-sharpe

        When Sharpe > 1, GOOD risk-adjusted returns

        When Sharpe > 2, VERY GOOD risk-adjusted returns

        When Sharpe > 3, EXCELLENT risk-adjusted returns

        When beta = 0, it means that there's no relationship.

        When beta < 1, it means that the stock is defensive (less prone to high highs and low lows)

        When beta > 1, it means that the stock is aggresive (more prone to high highs and low lows)

        Params
        ------
        data: series of stock price data
        mark_ticker: ticker of the market data you want to compute CAPM metrics with (default is ^GSPC)
        start: data from which to download data (default Jan 1st 2010)
        riskfree: the assumed risk free yield (US 10 Year Bond is assumed: 2.5%)

        Returns
        -------
        Dataframe with CAPM metrics computed against specified market procy
        """
        # Beta
        dd, mark_ret = self._market_data_combination(data, mark_ticker, start_time)
        log_ret = self.log_returns(dd)
        covar = log_ret.cov() * 252
        covar = pd.DataFrame(covar.iloc[:-1, -1])
        mrk_var = log_ret.iloc[:, -1].var() * 252
        beta = covar / mrk_var

        stdev_ret = pd.DataFrame(((log_ret.std() * 250 ** 0.5)[:-1]), columns=["STD"])
        beta = beta.merge(stdev_ret, left_index=True, right_index=True)

        # CAPM
        for i, row in beta.iterrows():
            beta.at[i, "CAPM"] = riskfree + (row[mark_ticker] * (mark_ret - riskfree))
        # Sharpe
        for i, row in beta.iterrows():
            beta.at[i, "Sharpe"] = (row["CAPM"] - riskfree) / (row["STD"])
        beta.rename(columns={mark_ticker: "Beta"}, inplace=True)
        logger.info(beta)
        return beta

    def drift_calc(self, data: pd.Series, return_type: str = "log"):
        """Calculate the drift of data"""
        if return_type == "log":
            lr = self.log_returns(data)
        elif return_type == "simple":
            lr = self.simple_returns(data)
        u = lr.mean()
        var = lr.var()
        drift = u - (0.5 * var)
        try:
            return drift.values
        except:
            return drift

    def daily_returns(
        self,
        data: pd.Series,
        period: int = 252,
        iterations: int = 1000,
        return_type="log",
    ):
        """Calculate the daily returns"""
        ft = self.drift_calc(data, return_type)
        if return_type == "log":
            try:
                stv = self.log_returns(data).std().values
            except:
                stv = self.log_returns(data).std()
        elif return_type == "simple":
            try:
                stv = self.simple_returns(data).std().values
            except:
                stv = self.simple_returns(data).std()
        # Oftentimes, we find that the distribution of returns is a variation of the normal distribution where it has a fat tail
        # This distribution is called cauchy distribution
        dr = np.exp(ft + stv * norm.ppf(np.random.rand(period, iterations)))
        return dr

    def probs_find(self, predicted: float, higherthan: float, on: str = "value"):
        """
        This function calculated the probability of a stock being above a certain threshold, which can be defined as a value (final stock price) or return rate (percentage change)

        Params
        ------

        1. predicted: dataframe with all the predicted prices (days and simulations)
        2. higherthan: specified threshhold to which compute the probability (ex. 0 on return will compute the probability of at least breakeven)
        3. on: 'return' or 'value', the return of the stock or the final value of stock for every simulation over the time specified
        """
        if on == "return":
            predicted0 = predicted.iloc[0, 0]
            predicted = predicted.iloc[-1]
            predList = list(predicted)
            over = [
                (i * 100) / predicted0
                for i in predList
                if ((i - predicted0) * 100) / predicted0 >= higherthan
            ]
            less = [
                (i * 100) / predicted0
                for i in predList
                if ((i - predicted0) * 100) / predicted0 < higherthan
            ]
        elif on == "value":
            predicted = predicted.iloc[-1]
            predList = list(predicted)
            over = [i for i in predList if i >= higherthan]
            less = [i for i in predList if i < higherthan]
        else:
            logger.info("'on' must be either value or return")
        return len(over) / (len(over) + len(less))

    def simulate(
        self,
        data: pd.Series,
        period: int = 252,
        iterations: int = 1000,
        return_type: str = "log",
        plot: bool = False,
        loud: bool = True,
    ):
        """"""
        # Generate daily returns
        returns = self.daily_returns(data, period, iterations, return_type)
        # Create empty matrix
        price_list = np.zeros_like(returns)
        # Put the last actual price in the first row of matrix.
        price_list[0] = data.iloc[-1]
        # Calculate the price of each day
        for t in range(1, period):
            price_list[t] = price_list[t - 1] * returns[t]

        # Plot Option
        if plot == True:
            x = pd.DataFrame(price_list).iloc[-1]
            fig, ax = plt.subplots(1, 2, figsize=(14, 4))
            sns.distplot(x, ax=ax[0])
            sns.distplot(
                x, hist_kws={"cumulative": True}, kde_kws={"cumulative": True}, ax=ax[1]
            )
            plt.xlabel("Stock Price")
            plt.show()

        # CAPM and Sharpe Ratio
        pob = self.probs_find(pd.DataFrame(price_list), 0, on="return")
        ret = (
            100
            * (pd.DataFrame(price_list).iloc[-1].mean() - price_list[0, 1])
            / pd.DataFrame(price_list).iloc[-1].mean()
        )
        expected_value = round(pd.DataFrame(price_list).iloc[-1].mean(), 2)
        # logger.infoing information about stock
        if loud:
            try:
                out_string = [nam for nam in data.columns][0]
            except:
                out_string = data.name
            out_string += (
                (f"\nDays: {period-1}" + f"\nExpected Value: ${expected_value}")
                + f"\nReturn: {round(ret,2)}%"
                + f"\nProbability of Breakeven: {pob}"
            )
            logger.info(out_string)
            return pd.DataFrame(price_list), pob


if __name__ == "__main__":
    mc = MonteCarlo()
    data = mc.alpaca.import_stock_data(["pg"])
    mc.simulate(data, 252, 5000, "log", plot=True)
