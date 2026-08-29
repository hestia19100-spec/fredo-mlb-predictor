import unittest

from src.retry_policy import (
    RetryPolicy,
    run_with_retries,
)


class TransientTestError(RuntimeError):
    """Erreur temporaire utilisée dans les tests."""


class PermanentTestError(RuntimeError):
    """Erreur qui ne doit jamais être retentée."""


class RetryPolicyTests(unittest.TestCase):
    """Vérifie les tentatives et les temporisations contrôlées."""

    def test_success_on_first_attempt_does_not_wait(self) -> None:
        """Une réussite immédiate ne doit provoquer aucune attente."""
        calls = 0
        delays: list[float] = []

        def operation() -> str:
            nonlocal calls
            calls += 1
            return "succès"

        outcome = run_with_retries(
            operation,
            retry_exceptions=(TransientTestError,),
            sleep_function=delays.append,
        )

        self.assertEqual(outcome.value, "succès")
        self.assertEqual(outcome.attempts, 1)
        self.assertEqual(calls, 1)
        self.assertEqual(delays, [])

    def test_transient_failures_are_retried_with_backoff(
        self,
    ) -> None:
        """Deux erreurs temporaires doivent attendre 1 puis 2 secondes."""
        calls = 0
        delays: list[float] = []

        def operation() -> str:
            nonlocal calls
            calls += 1

            if calls < 3:
                raise TransientTestError(
                    f"Erreur temporaire n° {calls}"
                )

            return "récupéré"

        outcome = run_with_retries(
            operation,
            retry_exceptions=(TransientTestError,),
            policy=RetryPolicy(
                max_attempts=3,
                initial_delay_seconds=1.0,
                backoff_multiplier=2.0,
            ),
            sleep_function=delays.append,
        )

        self.assertEqual(outcome.value, "récupéré")
        self.assertEqual(outcome.attempts, 3)
        self.assertEqual(calls, 3)
        self.assertEqual(delays, [1.0, 2.0])

    def test_last_transient_error_is_raised(self) -> None:
        """La dernière erreur doit remonter après épuisement."""
        calls = 0
        delays: list[float] = []

        def operation() -> None:
            nonlocal calls
            calls += 1
            raise TransientTestError(
                f"Échec définitif après tentative {calls}"
            )

        with self.assertRaises(TransientTestError):
            run_with_retries(
                operation,
                retry_exceptions=(TransientTestError,),
                policy=RetryPolicy(
                    max_attempts=3,
                    initial_delay_seconds=0.5,
                    backoff_multiplier=2.0,
                ),
                sleep_function=delays.append,
            )

        self.assertEqual(calls, 3)
        self.assertEqual(delays, [0.5, 1.0])

    def test_unlisted_error_is_not_retried(self) -> None:
        """Une erreur non autorisée doit remonter immédiatement."""
        calls = 0
        delays: list[float] = []

        def operation() -> None:
            nonlocal calls
            calls += 1
            raise PermanentTestError("Erreur permanente")

        with self.assertRaises(PermanentTestError):
            run_with_retries(
                operation,
                retry_exceptions=(TransientTestError,),
                policy=RetryPolicy(max_attempts=5),
                sleep_function=delays.append,
            )

        self.assertEqual(calls, 1)
        self.assertEqual(delays, [])

    def test_invalid_policies_are_rejected(self) -> None:
        """Les paramètres dangereux doivent être refusés."""
        invalid_policy_factories = (
            lambda: RetryPolicy(max_attempts=0),
            lambda: RetryPolicy(max_attempts=True),
            lambda: RetryPolicy(initial_delay_seconds=-1),
            lambda: RetryPolicy(initial_delay_seconds=True),
            lambda: RetryPolicy(backoff_multiplier=0.5),
            lambda: RetryPolicy(backoff_multiplier=True),
        )

        for policy_factory in invalid_policy_factories:
            with self.subTest(policy_factory=policy_factory):
                with self.assertRaises(ValueError):
                    policy_factory()

    def test_invalid_retry_exception_lists_are_rejected(
        self,
    ) -> None:
        """La liste des erreurs à retenter doit être explicite."""
        with self.assertRaises(ValueError):
            run_with_retries(
                lambda: "résultat",
                retry_exceptions=(),
            )

        with self.assertRaises(ValueError):
            run_with_retries(
                lambda: "résultat",
                retry_exceptions=(str,),
            )


if __name__ == "__main__":
    unittest.main()