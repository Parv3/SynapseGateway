import time
from enum import Enum
import logging
from typing import Callable, Any

logger = logging.getLogger("synapse.circuit_breaker")

class CircuitState(str, Enum):
    CLOSED = "CLOSED"       # Normal flow
    OPEN = "OPEN"           # Tripped; do not call provider
    HALF_OPEN = "HALF_OPEN" # Testing recovery with a single probe

class CircuitBreaker:
    """
    Finite State Machine implementing the Circuit Breaker pattern.
    Protects latency and instantly redirects traffic during upstream outages.
    """
    def __init__(
        self,
        name: str,
        failure_threshold: int = 3,
        recovery_time_sec: float = 30.0
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_time_sec = recovery_time_sec
        
        self.state: CircuitState = CircuitState.CLOSED
        self.failure_count: int = 0
        self.last_state_change: float = time.time()
        self.last_failure_time: float = 0.0
        self._probe_in_flight: bool = False

    def can_execute(self) -> bool:
        """
        Returns True if request can be attempted, False if circuit is OPEN.
        Enforces a single probe request when in HALF_OPEN state.
        """
        now = time.time()
        if self.state == CircuitState.OPEN:
            if now - self.last_failure_time >= self.recovery_time_sec:
                logger.info(f"CircuitBreaker [{self.name}] transitioning OPEN -> HALF_OPEN (Probing recovery)")
                self.state = CircuitState.HALF_OPEN
                self.last_state_change = now
                self._probe_in_flight = True
                return True
            return False

        if self.state == CircuitState.HALF_OPEN:
            # Allow only the single designated probe through; others fail over until probe succeeds
            if not self._probe_in_flight:
                self._probe_in_flight = True
                return True
            return False

        return True

    def record_success(self) -> None:
        """Records a successful request, resetting failures."""
        if self.state == CircuitState.HALF_OPEN:
            logger.info(f"CircuitBreaker [{self.name}] probe succeeded. Transitioning HALF_OPEN -> CLOSED")
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self._probe_in_flight = False

    def record_failure(self) -> None:
        """Records a failure and trips the breaker if threshold is reached."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        self._probe_in_flight = False

        if self.state == CircuitState.HALF_OPEN or self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            self.last_state_change = time.time()
            logger.warning(
                f"CircuitBreaker [{self.name}] TRIPPED to OPEN! "
                f"Failures: {self.failure_count}/{self.failure_threshold}. Will pause for {self.recovery_time_sec}s."
            )

    def status(self) -> dict:
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "last_failure_age_sec": round(time.time() - self.last_failure_time, 1) if self.last_failure_time else None
        }
