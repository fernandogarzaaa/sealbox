"""Per-session leakage budget: epsilon (differential privacy) + bytes of released information."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

from .policy import SessionConfig


@dataclass
class BudgetDecision:
    ok: bool
    reason: str = ""


@dataclass
class SessionBudget:
    config: SessionConfig
    epsilon_spent: float = 0.0
    bytes_released: int = 0
    queries: int = 0
    errors: int = 0
    data_errors: int = 0
    denials: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def admit_query(self) -> BudgetDecision:
        """Called before any compute. Counts the query even if it is later denied."""
        with self._lock:
            if self.queries >= self.config.max_queries:
                return BudgetDecision(False, f"query limit reached ({self.config.max_queries} per session)")
            if self.errors >= self.config.max_errors:
                return BudgetDecision(
                    False, f"error limit reached ({self.config.max_errors}); errors can be used as an oracle"
                )
            if self.data_errors >= self.config.max_data_errors:
                return BudgetDecision(
                    False,
                    f"data-dependent error limit reached ({self.config.max_data_errors}); runtime errors that "
                    "depend on protected values are an oracle",
                )
            if self.bytes_released >= self.config.bytes_budget:
                return BudgetDecision(False, "bytes-of-information budget exhausted")
            self.queries += 1
            return BudgetDecision(True)

    def can_spend_epsilon(self, eps: float) -> BudgetDecision:
        with self._lock:
            if self.epsilon_spent + eps > self.config.epsilon_budget + 1e-12:
                return BudgetDecision(
                    False,
                    f"privacy budget exhausted (epsilon spent {self.epsilon_spent:.2f} of "
                    f"{self.config.epsilon_budget:.2f}; this query needs {eps:.2f})",
                )
            return BudgetDecision(True)

    def charge(self, nbytes: int, eps: float = 0.0) -> BudgetDecision:
        with self._lock:
            if self.bytes_released + nbytes > self.config.bytes_budget:
                return BudgetDecision(
                    False,
                    f"bytes-of-information budget would be exceeded ({self.bytes_released} + {nbytes} > "
                    f"{self.config.bytes_budget})",
                )
            if self.epsilon_spent + eps > self.config.epsilon_budget + 1e-12:
                return BudgetDecision(False, "privacy budget exhausted")
            self.bytes_released += nbytes
            self.epsilon_spent += eps
            return BudgetDecision(True)

    def record_error(self) -> None:
        with self._lock:
            self.errors += 1

    def record_data_error(self) -> None:
        with self._lock:
            self.data_errors += 1
            self.errors += 1

    def record_denial(self) -> None:
        with self._lock:
            self.denials += 1

    def status(self) -> dict[str, object]:
        c = self.config
        return {
            "epsilon": {
                "spent": round(self.epsilon_spent, 4),
                "budget": c.epsilon_budget,
                "remaining": round(max(0.0, c.epsilon_budget - self.epsilon_spent), 4),
            },
            "bytes": {
                "released": self.bytes_released,
                "budget": c.bytes_budget,
                "remaining": max(0, c.bytes_budget - self.bytes_released),
            },
            "queries": {"used": self.queries, "limit": c.max_queries},
            "errors": {"used": self.errors, "limit": c.max_errors},
            "data_errors": {"used": self.data_errors, "limit": c.max_data_errors},
            "denials": self.denials,
        }
