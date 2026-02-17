"""
Base agent interface for the News Trading System.

All concrete agents must inherit from BaseAgent and implement:
    - name  (property) – a human-readable identifier
    - run() – the agent's primary execution method

This contract allows the Coordinator to treat agents polymorphically and
makes each agent independently testable by mocking run().
"""

from abc import ABC, abstractmethod
from typing import Any


class BaseAgent(ABC):
    """Abstract base class that every agent in the system must extend."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name used in logs and progress output."""

    @abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> Any:
        """
        Execute the agent's primary task.

        Concrete implementations define the specific signature and return type
        appropriate for their responsibility.
        """
