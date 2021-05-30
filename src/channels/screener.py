from bs4 import BeautifulSoup
import requests, logging
import pandas as pd

logger = logging.getLogger(__name__)


class Screener:
    """Gets the most actve stocks"""

    def __init__(self, url: str = "https://finance.yahoo.com/most-active") -> None:
        self.url = url
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.93 Safari/537.36"
        }

    def _human_to_number(self, number: str):
        """Converts a number like '1.7M' to 1700000"""
        try:
            m = {"K": 3, "M": 6, "B": 9, "T": 12}
            return int(float(number[:-1]) * 10 ** m[number[-1]])
        except:
            return number

    def get_most_active(self, count: int = 100):
        """Pulls the most active stocks and returns a dataframe"""
        logger.info("Fetching most active stocks from yfinance")
        response = requests.get(
            self.url + f"?offset=0&count={count}", headers=self.headers
        )
        soup = BeautifulSoup(response.content, "lxml")
        items = []
        for row in (
            soup.find("table", attrs={"class": "W(100%)"}).find("tbody").find_all("tr")
        ):
            cols = row.find_all("td")
            cols = [ele.text.strip() for ele in cols]
            item = {
                "symbol": cols[0],
                "name": cols[1],
                "price": cols[2],
                "change": cols[3],
                "pct_change": cols[4],
                "volume": self._human_to_number(cols[5]),
                "avg_volume": self._human_to_number(cols[6]),
                "cap": self._human_to_number(cols[7]),
                "pe_ratio": cols[8],
            }
            items.append(item)
        if len(items) > 0:
            return pd.DataFrame(items)
        else:
            logger.warning("Something went wrong")
            return pd.DataFrame()


if __name__ == "__main__":
    screener = Screener()
    print(screener.get_most_active())
