"""Abstract base service enforcing SRP and DIP."""

from abc import ABC, abstractmethod


class BaseService(ABC):
    """All services inherit from this base to maintain a consistent interface."""
    pass
