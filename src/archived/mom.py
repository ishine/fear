from statistics import mean

import numpy as np
import pandas as pd
from tqdm import tqdm
from strategies.base import BaseStrategy


class MOMStrategy(BaseStrategy):
    def __init__(
        self,
        lags: int = 7,
        cols: list = ["return"],
    ) -> None:
        super(MOMStrategy, self).__init__(lags=lags, cols=cols)

    def _create_features(self, data: pd.DataFrame, i):
        data = data.copy()
        data["sup_tolerance"] = pd.Series(np.zeros(len(data)))
        data["res_tolerance"] = pd.Series(np.zeros(len(data)))
        data["sup_count"] = pd.Series(np.zeros(len(data)))
        data["res_count"] = pd.Series(np.zeros(len(data)))
        data["sup"] = pd.Series(np.zeros(len(data)))
        data["res"] = pd.Series(np.zeros(len(data)))
        data["positions"] = pd.Series(np.zeros(len(data)))
        data["signal"] = pd.Series(np.zeros(len(data)))
        in_support = 0
        in_resistance = 0

        for x in range((i - 1) + i, len(data)):
            data_section = data[x - i : x + 1]
            support_level = min(data_section["close"])
            resistance_level = max(data_section["close"])
            range_level = resistance_level - support_level
            data["res"][x] = resistance_level
            data["sup"][x] = support_level
            data["sup_tolerance"][x] = support_level + 0.2 * range_level
            data["res_tolerance"][x] = resistance_level - 0.2 * range_level

            if (
                data["close"][x] >= data["res_tolerance"][x]
                and data["close"][x] <= data["res"][x]
            ):
                in_resistance += 1
                data["res_count"][x] = in_resistance
            elif (
                data["close"][x] <= data["sup_tolerance"][x]
                and data["close"][x] >= data["sup"][x]
            ):
                in_support += 1
                data["sup_count"][x] = in_support
            else:
                in_support = 0
                in_resistance = 0

            if in_resistance > 2:
                data["signal"][x] = 1
            elif in_support > 2:
                data["signal"][x] = 0
            else:
                data["signal"][x] = data["signal"][x - 1]

        data["positions"] = data["signal"].diff()
        return data

    def evaluate(self, data: pd.DataFrame):
        different_simulations = []
        i_range = [x for x in range(1, 30)]

        for i in tqdm(i_range):

            data = self._create_features(data, i)
            data["buy_sell"] = data["signal"].replace(0, -1)

            data["Orig_Returns"] = np.log(data["close"] / data["close"].shift(1))
            Cum_aapl_returns = data["Orig_Returns"].cumsum() * 100
            data["Strategy_Returns"] = data["Orig_Returns"] * data["buy_sell"].shift(1)
            Cum_strategy_returns = data["Strategy_Returns"].cumsum() * 100

            Cum_aapl_returns = Cum_aapl_returns.dropna()
            Cum_strategy_returns = Cum_strategy_returns.dropna()

            different_simulations.append(
                mean(int(x - y) for x, y in zip(Cum_strategy_returns, Cum_aapl_returns))
            )
        df_nb_k = pd.DataFrame(
            {
                "nb": i_range,
                "diff": different_simulations,
            }
        ).set_index("nb")
        return df_nb_k
