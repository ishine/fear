try:
    from channels.base import BaseAPI
except:
    from base import BaseAPI
from os import access
import tpqoa, logging, statistics
from datetime import datetime, timedelta
import pandas as pd
import fxcmpy

logging.basicConfig(
    # filename,
    format="[%(name)s] %(levelname)s %(asctime)s %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

DTFORMAT = "%Y-%m-%dT%H:%M:%SZ"


class FXCM(BaseAPI):
    def __init__(self, config: str = "keys/fxcm.cfg", mode: str = "demo") -> None:
        self.api = fxcmpy.fxcmpy(config_file=config, server=mode)

    def get_symbols(self):
        """Get available symbols"""
        try:
            return self.api.get_instruments()
        except Exception as e:
            logging.warning(f"Couldn't get instruments ({e})")
            return []

    def submit_market_order(self, symbol: str, side: str, qty: int):
        """Submit a market order"""
        try:
            if side == "buy":
                self.api.create_market_buy_order(symbol, qty)
            elif side == "sell":
                self.api.create_market_sell_order(symbol, qty)
            else:
                logger.warning(f"Side '{side}' not 'buy' or 'sell'")
            return True
        except Exception as e:
            logger.warning(
                f"Couldn't submit market {side} order for {qty} {symbol} ({e})"
            )
            return False

    def get_shares(self, symbol: str):
        """Returns the share count that you possess for a symbol"""
        try:
            pos = self.api.get_open_positions(symbol)
            return pos[pos["currency"] == symbol].sum()
        except Exception as e:  # effectively 0 then
            logger.warning(f"Couldn't get positions for {symbol} ({e})")
            return 0