from abc import abstractmethod


class BaseAPI:
    """Base class for the apis"""

    @abstractmethod
    def get_bars(self):
        """Get bars"""

    @abstractmethod
    def submit_limit_order(self):
        """Submit a limit order"""

    @abstractmethod
    def submit_market_order(self):
        """Submit a limit order"""

    @abstractmethod
    def get_account(self):
        """
        Get account info
        """

    @abstractmethod
    def any_open_orders(self) -> bool:
        """
        Returns true if any open orders
        """

    @abstractmethod
    def get_position(self, symbol):
        """Gets the position for a symbol"""

    @abstractmethod
    def get_shares(self, symbol):
        """Returns the share count that you possess for a symbol"""

    @abstractmethod
    def get_buying_power(self):
        """Returns the buying_power you have"""