"""Politique générique de nouvelles tentatives contrôlées."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import time
from typing import Generic, TypeVar


ResultType = TypeVar("ResultType")


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Décrit le nombre de tentatives et leur temporisation."""

    max_attempts: int = 3
    initial_delay_seconds: float = 1.0
    backoff_multiplier: float = 2.0

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_attempts, bool)
            or not isinstance(self.max_attempts, int)
            or self.max_attempts < 1
        ):
            raise ValueError(
                "Le nombre maximal de tentatives doit être "
                "un entier positif."
            )

        if (
            isinstance(self.initial_delay_seconds, bool)
            or not isinstance(
                self.initial_delay_seconds,
                (int, float),
            )
            or self.initial_delay_seconds < 0
        ):
            raise ValueError(
                "Le délai initial doit être un nombre positif "
                "ou nul."
            )

        if (
            isinstance(self.backoff_multiplier, bool)
            or not isinstance(
                self.backoff_multiplier,
                (int, float),
            )
            or self.backoff_multiplier < 1
        ):
            raise ValueError(
                "Le multiplicateur d’attente doit être "
                "supérieur ou égal à 1."
            )

    def delay_after_failure(
        self,
        failed_attempt_number: int,
    ) -> float:
        """Calcule l’attente avant la tentative suivante."""
        if (
            isinstance(failed_attempt_number, bool)
            or not isinstance(failed_attempt_number, int)
            or not 1 <= failed_attempt_number < self.max_attempts
        ):
            raise ValueError(
                "Le numéro de tentative échouée ne permet pas "
                "une nouvelle tentative."
            )

        return float(
            self.initial_delay_seconds
            * (
                self.backoff_multiplier
                ** (failed_attempt_number - 1)
            )
        )


@dataclass(frozen=True, slots=True)
class RetryOutcome(Generic[ResultType]):
    """Conserve le résultat et le nombre de tentatives utilisées."""

    value: ResultType
    attempts: int


def run_with_retries(
    operation: Callable[[], ResultType],
    *,
    retry_exceptions: tuple[type[Exception], ...],
    policy: RetryPolicy = RetryPolicy(),
    sleep_function: Callable[[float], None] = time.sleep,
) -> RetryOutcome[ResultType]:
    """Exécute une opération et retente uniquement les erreurs prévues."""
    if not retry_exceptions:
        raise ValueError(
            "Au moins une catégorie d’erreur à retenter est requise."
        )

    if any(
        not isinstance(exception_type, type)
        or not issubclass(exception_type, Exception)
        for exception_type in retry_exceptions
    ):
        raise ValueError(
            "Les erreurs à retenter doivent être des exceptions."
        )

    for attempt_number in range(1, policy.max_attempts + 1):
        try:
            value = operation()
        except retry_exceptions:
            if attempt_number >= policy.max_attempts:
                raise

            delay_seconds = policy.delay_after_failure(
                attempt_number
            )
            sleep_function(delay_seconds)
        else:
            return RetryOutcome(
                value=value,
                attempts=attempt_number,
            )

    raise RuntimeError(
        "La boucle de nouvelles tentatives s’est terminée "
        "de façon inattendue."
    )