from abc import abstractmethod


class BaseAPI:
    """Base class for the apis"""

    @abstractmethod
    def get_bars(self):
        """Get bars"""

    @abstractmethod
    def submit_limit_order(self):
        """Submit a limit order"""