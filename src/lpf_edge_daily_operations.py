"""Controle explicite des futures actions quotidiennes LPF Edge.

Les fonctions d'inspection restent strictement locales et sans effet de bord.
La collecte MLB n'est possible que par ``refresh_daily_mlb_data`` : cette
fonction publique exige la date du jour et ne lance ni modele, ni
certification, ni scoring.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from enum import Enum
import os
from pathlib import Path
import sqlite3
import subprocess
from typing import Sequence
from zoneinfo import ZoneInfo

from src.database import DATA_DIR, DATABASE_PATH
from src.ingestion_service import (
    ScheduleIngestionResult,
    run_schedule_ingestion,
)
from src.lpf_edge_dashboard import (
    LPFEdgeDashboardError,
    list_certified_prediction_dates,
    load_certified_prediction_day,
    load_latest_score_summary,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PARIS_TIMEZONE = ZoneInfo("Europe/Paris")
MINIMUM_PREDICTION_LEAD_MINUTES = 120
SCORING_CHECKPOINT_TIME_UTC = time(6, 0, 0)
PREDICTION_ROOT = Path(
    "shadow_results/logistic_team_form_v1_platt_shadow_v2"
)
CERTIFICATION_ROOT = Path(
    "shadow_certifications/logistic_team_form_v1_platt_shadow_v2"
)
SCORING_ROOT = Path(
    "shadow_scores/logistic_team_form_v1_platt_shadow_v2_2026_v1"
)
PREDICTION_SUCCESS_FILENAMES = (
    "COMPLETED",
    "RESERVED",
    "activation_reverification.remote.json.gz",
    "candidate_ledger.csv",
    "features.csv",
    "predictions.csv",
    "receipt.json",
    "source_snapshot.json.gz",
)
_FINAL_STATUS_CODES = frozenset({"F", "FG", "FO", "FR"})
_FINAL_STATUS_DETAILS = frozenset(
    {"FINAL", "GAME OVER", "COMPLETED EARLY"}
)


class DailyOperationsError(RuntimeError):
    """Etat local impossible a etablir sans ambiguite."""


class DailyActionState(str, Enum):
    """Etats fermes affichables par le futur centre d'actions."""

    READY = "READY"
    DONE = "DONE"
    NEED_DATA = "NEED_DATA"
    TOO_EARLY = "TOO_EARLY"
    TOO_LATE = "TOO_LATE"
    ACTION_REQUIRED = "ACTION_REQUIRED"
    BLOCKED = "BLOCKED"
    NOT_AVAILABLE = "NOT_AVAILABLE"


class PredictionSlotState(str, Enum):
    """Etat local minimal du dossier de prediction du jour."""

    ABSENT = "ABSENT"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    PARTIAL = "PARTIAL"


class DailyPredictionStage(str, Enum):
    """Etape exacte atteinte par l'automatisation des predictions."""

    PREFLIGHT = "PREFLIGHT"
    PREDICTION = "PREDICTION"
    RESULTS_PUBLICATION = "RESULTS_PUBLICATION"
    CERTIFICATION = "CERTIFICATION"
    CERTIFICATION_PUBLICATION = "CERTIFICATION_PUBLICATION"


class DailyPredictionAutomationError(DailyOperationsError):
    """Echec ferme de la chaine prediction, GitHub et certification."""

    def __init__(self, stage: DailyPredictionStage, message: str) -> None:
        self.stage = stage
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class GitWorkspaceState:
    """Etat Git local lu sans acces reseau et sans ecriture."""

    available: bool
    branch: str | None
    head_commit: str | None
    origin_main_commit: str | None
    clean: bool
    synchronized: bool

    @property
    def ready_for_publication(self) -> bool:
        return (
            self.available
            and self.branch == "main"
            and self.head_commit is not None
            and self.origin_main_commit is not None
            and self.clean
            and self.synchronized
        )


@dataclass(frozen=True, slots=True)
class LocalGameDayState:
    """Resume strictement local des matchs deja presents dans SQLite."""

    game_count: int
    final_game_count: int
    first_start_utc: datetime | None


@dataclass(frozen=True, slots=True)
class DailyAction:
    """Une action future et son autorisation locale courante."""

    key: str
    label: str
    state: DailyActionState
    message: str
    can_execute: bool

    def __post_init__(self) -> None:
        if self.can_execute != (self.state is DailyActionState.READY):
            raise ValueError("Seul un etat READY peut autoriser une action.")


@dataclass(frozen=True, slots=True)
class DailyOperationsOverview:
    """Vue complete qui alimentera les futurs boutons Streamlit."""

    target_date: date
    previous_date: date
    checkpoint_date: date
    inspected_at_utc: datetime
    game_day: LocalGameDayState
    git: GitWorkspaceState
    prediction_slot_state: PredictionSlotState
    prediction_certified: bool
    latest_score_pending_count: int | None
    data_action: DailyAction
    prediction_action: DailyAction
    results_action: DailyAction
    integrity_errors: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DailyPredictionPublication:
    """Recu final d'une prediction publiee et certifiee sur GitHub."""

    target_date: date
    batch_id: str
    results_commit: str
    certification_commit: str
    results_paths: tuple[str, ...]
    certification_paths: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _PredictionEngineOutcome:
    slot_path: Path
    batch_id: str
    batch_status: str
    receipt_sha256: str


@dataclass(frozen=True, slots=True)
class _CertificationOutcome:
    batch_id: str
    results_commit: str
    certification_path: Path
    evidence_path: Path


def _action(
    key: str,
    label: str,
    state: DailyActionState,
    message: str,
) -> DailyAction:
    return DailyAction(
        key=key,
        label=label,
        state=state,
        message=message,
        can_execute=state is DailyActionState.READY,
    )


def _prediction_action(
    *,
    target_date: date,
    paris_today: date,
    now_utc: datetime,
    game_day: LocalGameDayState,
    git_ready: bool,
    slot_state: PredictionSlotState,
    certified: bool,
    integrity_errors: tuple[str, ...],
) -> DailyAction:
    label = "Créer les prédictions du jour"
    if integrity_errors:
        return _action(
            "predictions",
            label,
            DailyActionState.BLOCKED,
            "Un contrôle d’intégrité local doit être résolu avant toute prédiction.",
        )
    if target_date != paris_today:
        return _action(
            "predictions",
            label,
            DailyActionState.NOT_AVAILABLE,
            "Les prédictions officielles peuvent seulement être créées pour aujourd’hui.",
        )
    if certified:
        return _action(
            "predictions",
            label,
            DailyActionState.DONE,
            "Les prédictions du jour sont déjà créées, publiées et certifiées.",
        )
    if slot_state is PredictionSlotState.COMPLETED:
        return _action(
            "predictions",
            label,
            DailyActionState.ACTION_REQUIRED,
            "Le lot est créé mais sa publication et sa certification restent à terminer.",
        )
    if slot_state is PredictionSlotState.FAILED:
        return _action(
            "predictions",
            label,
            DailyActionState.BLOCKED,
            "La tentative du jour a échoué et son dossier est définitivement conservé.",
        )
    if slot_state is PredictionSlotState.PARTIAL:
        return _action(
            "predictions",
            label,
            DailyActionState.BLOCKED,
            "Une tentative partielle existe déjà ; elle ne peut être ni reprise ni écrasée.",
        )
    if not git_ready:
        return _action(
            "predictions",
            label,
            DailyActionState.BLOCKED,
            "Le dépôt doit être propre, sur main et synchronisé avant la prédiction.",
        )
    if game_day.game_count == 0:
        return _action(
            "predictions",
            label,
            DailyActionState.NEED_DATA,
            "Récupère d’abord les matchs du jour depuis MLB.",
        )
    if game_day.first_start_utc is None:
        return _action(
            "predictions",
            label,
            DailyActionState.BLOCKED,
            "L’heure du premier match est absente ou invalide.",
        )
    lead_minutes = (game_day.first_start_utc - now_utc).total_seconds() / 60
    if lead_minutes < MINIMUM_PREDICTION_LEAD_MINUTES:
        return _action(
            "predictions",
            label,
            DailyActionState.TOO_LATE,
            "Le délai obligatoire de deux heures avant le premier match est dépassé.",
        )
    return _action(
        "predictions",
        label,
        DailyActionState.READY,
        f"Prêt avec {lead_minutes / 60:.1f} h d’avance avant le premier match.",
    )


def _results_action(
    *,
    target_date: date,
    paris_today: date,
    now_utc: datetime,
    previous_certified: bool,
    latest_pending_count: int | None,
    checkpoint_slot_exists: bool,
    git_ready: bool,
    integrity_errors: tuple[str, ...],
) -> DailyAction:
    label = "Récupérer et vérifier les résultats"
    if integrity_errors:
        return _action(
            "results",
            label,
            DailyActionState.BLOCKED,
            "Un contrôle d’intégrité local doit être résolu avant le scoring.",
        )
    if target_date != paris_today:
        return _action(
            "results",
            label,
            DailyActionState.NOT_AVAILABLE,
            "Le centre quotidien doit être positionné sur la date d’aujourd’hui.",
        )
    if not previous_certified:
        return _action(
            "results",
            label,
            DailyActionState.NOT_AVAILABLE,
            "Aucune prédiction certifiée de la veille n’est à vérifier.",
        )
    if latest_pending_count == 0:
        return _action(
            "results",
            label,
            DailyActionState.DONE,
            "Toutes les prédictions certifiées de la veille sont déjà vérifiées.",
        )
    checkpoint_instant = datetime.combine(
        target_date,
        SCORING_CHECKPOINT_TIME_UTC,
        tzinfo=timezone.utc,
    )
    if now_utc < checkpoint_instant:
        return _action(
            "results",
            label,
            DailyActionState.TOO_EARLY,
            "Le contrôle officiel des résultats ouvre à 08:00, heure de Paris.",
        )
    if checkpoint_slot_exists:
        return _action(
            "results",
            label,
            DailyActionState.BLOCKED,
            "Le contrôle prévu aujourd’hui est déjà consommé et ne peut être relancé.",
        )
    if not git_ready:
        return _action(
            "results",
            label,
            DailyActionState.BLOCKED,
            "Le dépôt doit être propre, sur main et synchronisé avant le scoring.",
        )
    return _action(
        "results",
        label,
        DailyActionState.READY,
        "Le contrôle officiel des résultats de la veille peut être lancé.",
    )


def build_daily_operations_overview(
    *,
    target_date: date,
    paris_today: date,
    now_utc: datetime,
    game_day: LocalGameDayState,
    git: GitWorkspaceState,
    prediction_slot_state: PredictionSlotState,
    prediction_certified: bool,
    previous_certified: bool,
    latest_score_pending_count: int | None,
    checkpoint_slot_exists: bool,
    integrity_errors: Sequence[str] = (),
) -> DailyOperationsOverview:
    """Construit les trois decisions sans aucun acces disque ou reseau."""
    if not isinstance(target_date, date) or isinstance(target_date, datetime):
        raise TypeError("target_date doit être une date exacte.")
    if not isinstance(paris_today, date) or isinstance(paris_today, datetime):
        raise TypeError("paris_today doit être une date exacte.")
    if not isinstance(now_utc, datetime) or now_utc.tzinfo is None:
        raise TypeError("now_utc doit être un instant UTC conscient.")
    normalized_now = now_utc.astimezone(timezone.utc)
    if normalized_now != now_utc:
        now_utc = normalized_now
    errors = tuple(str(item) for item in integrity_errors)
    if any(not item for item in errors):
        raise ValueError("Une erreur d’intégrité ne peut pas être vide.")
    if latest_score_pending_count is not None and latest_score_pending_count < 0:
        raise ValueError("Le nombre de résultats en attente est invalide.")
    data_message = (
        f"{game_day.game_count} match(s) en base ; une actualisation MLB est possible."
        if game_day.game_count
        else "Aucun match en base ; une récupération MLB est nécessaire."
    )
    data_action = _action(
        "data",
        "Récupérer ou actualiser les données MLB",
        DailyActionState.READY,
        data_message,
    )
    prediction_action = _prediction_action(
        target_date=target_date,
        paris_today=paris_today,
        now_utc=now_utc,
        game_day=game_day,
        git_ready=git.ready_for_publication,
        slot_state=prediction_slot_state,
        certified=prediction_certified,
        integrity_errors=errors,
    )
    results_action = _results_action(
        target_date=target_date,
        paris_today=paris_today,
        now_utc=now_utc,
        previous_certified=previous_certified,
        latest_pending_count=latest_score_pending_count,
        checkpoint_slot_exists=checkpoint_slot_exists,
        git_ready=git.ready_for_publication,
        integrity_errors=errors,
    )
    return DailyOperationsOverview(
        target_date=target_date,
        previous_date=target_date - timedelta(days=1),
        checkpoint_date=now_utc.date(),
        inspected_at_utc=now_utc,
        game_day=game_day,
        git=git,
        prediction_slot_state=prediction_slot_state,
        prediction_certified=prediction_certified,
        latest_score_pending_count=latest_score_pending_count,
        data_action=data_action,
        prediction_action=prediction_action,
        results_action=results_action,
        integrity_errors=errors,
    )


def _run_git(
    project_directory: Path,
    arguments: Sequence[str],
) -> str:
    environment = os.environ.copy()
    environment["GIT_OPTIONAL_LOCKS"] = "0"
    completed = subprocess.run(
        ["git", *arguments],
        cwd=project_directory,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    return completed.stdout.strip()


def inspect_git_workspace(
    *,
    project_directory: Path = PROJECT_ROOT,
) -> GitWorkspaceState:
    """Relit main, HEAD, origin/main et la proprete sans contacter GitHub."""
    try:
        branch = _run_git(project_directory, ["branch", "--show-current"])
        head = _run_git(project_directory, ["rev-parse", "HEAD"])
        origin = _run_git(project_directory, ["rev-parse", "origin/main"])
        status = _run_git(
            project_directory,
            ["status", "--porcelain=v1", "--untracked-files=all"],
        )
    except (OSError, subprocess.SubprocessError, UnicodeError):
        return GitWorkspaceState(False, None, None, None, False, False)
    return GitWorkspaceState(
        available=True,
        branch=branch,
        head_commit=head,
        origin_main_commit=origin,
        clean=status == "",
        synchronized=head == origin,
    )


def _parse_game_start(value: object) -> datetime:
    if type(value) is not str or not value:
        raise DailyOperationsError("Une heure de match locale est invalide.")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise DailyOperationsError("Une heure de match locale est invalide.") from error
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def load_local_game_day_state(
    target_date: date,
    *,
    database_path: Path = DATABASE_PATH,
) -> LocalGameDayState:
    """Lit SQLite en mode strictement read-only, sans creer de base absente."""
    if not database_path.is_file() or database_path.is_symlink():
        return LocalGameDayState(0, 0, None)
    uri = f"{database_path.resolve().as_uri()}?mode=ro"
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(uri, uri=True)
        rows = connection.execute(
            """
            SELECT game_datetime_utc, status_code, status_detail
            FROM games
            WHERE official_date = ?
            ORDER BY game_datetime_utc, game_id
            """,
            (target_date.isoformat(),),
        ).fetchall()
    except sqlite3.Error as error:
        raise DailyOperationsError(
            "La base locale des matchs ne peut pas être relue."
        ) from error
    finally:
        if connection is not None:
            connection.close()
    starts = [_parse_game_start(row[0]) for row in rows]
    final_count = sum(
        str(row[1] or "").strip().upper() in _FINAL_STATUS_CODES
        or str(row[2] or "").strip().upper() in _FINAL_STATUS_DETAILS
        for row in rows
    )
    return LocalGameDayState(
        game_count=len(rows),
        final_game_count=final_count,
        first_start_utc=min(starts) if starts else None,
    )


def inspect_prediction_slot(
    target_date: date,
    *,
    project_directory: Path = PROJECT_ROOT,
) -> PredictionSlotState:
    """Classe un dossier journalier sans accepter de lien symbolique."""
    slot = project_directory / PREDICTION_ROOT / target_date.isoformat()
    if not slot.exists():
        return PredictionSlotState.ABSENT
    if slot.is_symlink() or not slot.is_dir():
        return PredictionSlotState.PARTIAL
    try:
        names = {path.name for path in slot.iterdir()}
    except OSError:
        return PredictionSlotState.PARTIAL
    if "FAILED.json" in names:
        return PredictionSlotState.FAILED
    if "COMPLETED" in names:
        return PredictionSlotState.COMPLETED
    return PredictionSlotState.PARTIAL


def inspect_daily_operations(
    target_date: date | None = None,
    *,
    now_utc: datetime | None = None,
    project_directory: Path = PROJECT_ROOT,
    database_path: Path = DATABASE_PATH,
) -> DailyOperationsOverview:
    """Produit la vue locale complete sans reseau et sans mutation."""
    instant = now_utc or datetime.now(timezone.utc)
    if instant.tzinfo is None:
        raise DailyOperationsError("L’horloge du centre d’actions doit être en UTC.")
    instant = instant.astimezone(timezone.utc)
    paris_today = instant.astimezone(PARIS_TIMEZONE).date()
    selected = target_date or paris_today
    previous = selected - timedelta(days=1)
    errors: list[str] = []

    try:
        game_day = load_local_game_day_state(
            selected,
            database_path=database_path,
        )
    except DailyOperationsError as error:
        game_day = LocalGameDayState(0, 0, None)
        errors.append(str(error))
    git = inspect_git_workspace(project_directory=project_directory)
    slot_state = inspect_prediction_slot(
        selected,
        project_directory=project_directory,
    )

    try:
        certified_dates = list_certified_prediction_dates(
            project_directory=project_directory,
        )
    except OSError:
        certified_dates = []
        errors.append("Les certifications locales ne peuvent pas être inspectées.")
    certified = selected in certified_dates
    previous_certified = previous in certified_dates
    latest_pending: int | None = None
    try:
        if certified:
            load_certified_prediction_day(
                selected,
                project_directory=project_directory,
            )
        if previous_certified:
            previous_day = load_certified_prediction_day(
                previous,
                project_directory=project_directory,
            )
            summary = load_latest_score_summary(
                previous,
                certified_predictions=previous_day.predictions,
                project_directory=project_directory,
            )
            if summary is not None:
                latest_pending = summary.pending_count
    except LPFEdgeDashboardError as error:
        errors.append(str(error))

    checkpoint_slot = (
        project_directory
        / SCORING_ROOT
        / previous.isoformat()
        / "observations"
        / instant.date().isoformat()
    )
    checkpoint_exists = checkpoint_slot.exists()
    if checkpoint_exists and checkpoint_slot.is_symlink():
        errors.append("Le créneau de scoring local est un lien interdit.")

    return build_daily_operations_overview(
        target_date=selected,
        paris_today=paris_today,
        now_utc=instant,
        game_day=game_day,
        git=git,
        prediction_slot_state=slot_state,
        prediction_certified=certified,
        previous_certified=previous_certified,
        latest_score_pending_count=latest_pending,
        checkpoint_slot_exists=checkpoint_exists,
        integrity_errors=errors,
    )


def refresh_daily_mlb_data(
    target_date: date | None = None,
    *,
    now_utc: datetime | None = None,
    database_path: Path = DATABASE_PATH,
    data_directory: Path = DATA_DIR,
) -> ScheduleIngestionResult:
    """Recupere les matchs du jour par l'unique service auditable existant."""
    instant = now_utc or datetime.now(timezone.utc)
    if not isinstance(instant, datetime) or instant.tzinfo is None:
        raise DailyOperationsError("L'horloge de la collecte doit etre en UTC.")
    instant = instant.astimezone(timezone.utc)
    paris_today = instant.astimezone(PARIS_TIMEZONE).date()
    selected = target_date or paris_today
    if not isinstance(selected, date) or isinstance(selected, datetime):
        raise TypeError("target_date doit etre une date exacte.")
    if selected != paris_today:
        raise DailyOperationsError(
            "Le bouton quotidien peut seulement actualiser les matchs d'aujourd'hui."
        )
    return run_schedule_ingestion(
        start_date=selected,
        end_date=selected,
        database_path=database_path,
        data_directory=data_directory,
    )


def _run_git_mutation(
    project_directory: Path,
    arguments: Sequence[str],
    *,
    timeout_seconds: int = 120,
) -> str:
    """Execute une commande Git explicite, sans shell ni invite interactive."""
    environment = os.environ.copy()
    environment["GIT_TERMINAL_PROMPT"] = "0"
    completed = subprocess.run(
        ["git", *arguments],
        cwd=project_directory,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    return completed.stdout.strip()


def _git_paths(project_directory: Path, arguments: Sequence[str]) -> tuple[str, ...]:
    output = _run_git_mutation(project_directory, arguments)
    return tuple(line for line in output.splitlines() if line)


def _require_exact_publication_changes(
    project_directory: Path,
    expected_paths: tuple[str, ...],
    *,
    staged: bool,
    stage: DailyPredictionStage,
) -> None:
    expected = set(expected_paths)
    untracked = set(
        _git_paths(
            project_directory,
            ("ls-files", "--others", "--exclude-standard"),
        )
    )
    unstaged = set(
        _git_paths(project_directory, ("diff", "--name-only"))
    )
    cached = set(
        _git_paths(project_directory, ("diff", "--cached", "--name-only"))
    )
    valid = (
        not untracked
        and not unstaged
        and cached == expected
        if staged
        else untracked == expected and not unstaged and not cached
    )
    if not valid:
        raise DailyPredictionAutomationError(
            stage,
            "Les changements Git ne correspondent pas exactement aux preuves attendues.",
        )


def _commit_and_push_exact_paths(
    *,
    project_directory: Path,
    expected_paths: tuple[str, ...],
    expected_parent_commit: str,
    commit_message: str,
    stage: DailyPredictionStage,
) -> str:
    """Ajoute, commite et pousse uniquement une liste fermee de preuves."""
    try:
        branch = _run_git_mutation(
            project_directory,
            ("branch", "--show-current"),
        )
        head_before = _run_git_mutation(
            project_directory,
            ("rev-parse", "HEAD"),
        )
        remote_before = _run_git_mutation(
            project_directory,
            ("rev-parse", "origin/main"),
        )
        if (
            branch != "main"
            or head_before != expected_parent_commit
            or remote_before != expected_parent_commit
        ):
            raise DailyPredictionAutomationError(
                stage,
                "La base Git a change depuis le controle precedent.",
            )
        _require_exact_publication_changes(
            project_directory,
            expected_paths,
            staged=False,
            stage=stage,
        )
        _run_git_mutation(project_directory, ("add", "--", *expected_paths))
        _require_exact_publication_changes(
            project_directory,
            expected_paths,
            staged=True,
            stage=stage,
        )
        _run_git_mutation(project_directory, ("diff", "--cached", "--check"))
        _run_git_mutation(
            project_directory,
            ("commit", "-m", commit_message),
        )
        commit = _run_git_mutation(project_directory, ("rev-parse", "HEAD"))
        if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
            raise DailyPredictionAutomationError(
                stage,
                "Le commit produit par Git est invalide.",
            )
        _run_git_mutation(
            project_directory,
            ("push", "origin", "main"),
            timeout_seconds=180,
        )
        remote = _run_git_mutation(
            project_directory,
            ("rev-parse", "origin/main"),
        )
        status = _run_git_mutation(
            project_directory,
            ("status", "--porcelain=v1", "--untracked-files=all"),
        )
    except DailyPredictionAutomationError:
        raise
    except (OSError, subprocess.SubprocessError, UnicodeError) as error:
        raise DailyPredictionAutomationError(
            stage,
            "La publication GitHub a ete interrompue ; aucune nouvelle etape "
            "n'a ete lancee apres cet echec.",
        ) from error
    if remote != commit or status:
        raise DailyPredictionAutomationError(
            stage,
            "Le commit local n'est pas proprement synchronise avec origin/main.",
        )
    return commit


def _execute_prediction_engine(
    target_date: date,
    *,
    project_directory: Path,
    database_path: Path,
    data_directory: Path,
) -> _PredictionEngineOutcome:
    from src import shadow_prediction

    result = shadow_prediction.execute_shadow_prediction(
        target_date,
        project_directory=project_directory,
        database_path=database_path,
        data_directory=data_directory,
    )
    if type(result) is not shadow_prediction.ShadowCompletionPublication:
        raise DailyPredictionAutomationError(
            DailyPredictionStage.PREDICTION,
            "Le moteur n'a pas produit une nouvelle fermeture terminale exacte.",
        )
    return _PredictionEngineOutcome(
        slot_path=result.slot_path,
        batch_id=result.batch_id,
        batch_status=result.batch_status,
        receipt_sha256=result.receipt_sha256,
    )


def _certify_prediction(
    target_date: date,
    results_commit: str,
    *,
    project_directory: Path,
) -> _CertificationOutcome:
    from src import shadow_certification

    result = shadow_certification.certify_shadow_prediction(
        target_date.isoformat(),
        results_commit,
        project_directory=project_directory,
    )
    if type(result) is not shadow_certification.ShadowCertificationPublication:
        raise DailyPredictionAutomationError(
            DailyPredictionStage.CERTIFICATION,
            "Le service n'a pas produit une certification terminale exacte.",
        )
    return _CertificationOutcome(
        batch_id=result.batch_id,
        results_commit=result.results_commit,
        certification_path=result.certification_path,
        evidence_path=result.raw_evidence_path,
    )


def _french_date_label(value: date) -> str:
    months = (
        "janvier", "fevrier", "mars", "avril", "mai", "juin",
        "juillet", "aout", "septembre", "octobre", "novembre", "decembre",
    )
    return f"{value.day} {months[value.month - 1]} {value.year}"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def execute_daily_prediction_publication(
    target_date: date | None = None,
    *,
    project_directory: Path = PROJECT_ROOT,
    database_path: Path = DATABASE_PATH,
    data_directory: Path = DATA_DIR,
) -> DailyPredictionPublication:
    """Cree, pousse et certifie le lot du jour sans option de contournement."""
    instant = _utc_now()
    if not isinstance(instant, datetime) or instant.tzinfo is None:
        raise DailyPredictionAutomationError(
            DailyPredictionStage.PREFLIGHT,
            "L'horloge de la prediction doit etre un instant UTC valide.",
        )
    instant = instant.astimezone(timezone.utc)
    paris_today = instant.astimezone(PARIS_TIMEZONE).date()
    selected = target_date or paris_today
    if not isinstance(selected, date) or isinstance(selected, datetime):
        raise TypeError("target_date doit etre une date exacte.")
    if selected != paris_today:
        raise DailyPredictionAutomationError(
            DailyPredictionStage.PREFLIGHT,
            "Le moteur quotidien peut seulement traiter aujourd'hui.",
        )
    project = Path(project_directory).resolve(strict=True)
    overview = inspect_daily_operations(
        selected,
        now_utc=instant,
        project_directory=project,
        database_path=database_path,
    )
    if overview.prediction_action.state is not DailyActionState.READY:
        raise DailyPredictionAutomationError(
            DailyPredictionStage.PREFLIGHT,
            overview.prediction_action.message,
        )
    starting_commit = overview.git.head_commit
    if starting_commit is None:
        raise DailyPredictionAutomationError(
            DailyPredictionStage.PREFLIGHT,
            "Le commit Git de depart est absent.",
        )

    try:
        prediction = _execute_prediction_engine(
            selected,
            project_directory=project,
            database_path=database_path,
            data_directory=data_directory,
        )
    except DailyPredictionAutomationError:
        raise
    except Exception as error:
        raise DailyPredictionAutomationError(
            DailyPredictionStage.PREDICTION,
            "Le moteur de prediction s'est arrete en conservant ses preuves locales.",
        ) from error

    expected_slot = project / PREDICTION_ROOT / selected.isoformat()
    if prediction.slot_path.resolve() != expected_slot.resolve():
        raise DailyPredictionAutomationError(
            DailyPredictionStage.PREDICTION,
            "Le moteur a retourne un dossier de prediction inattendu.",
        )
    results_paths = tuple(
        (PREDICTION_ROOT / selected.isoformat() / name).as_posix()
        for name in PREDICTION_SUCCESS_FILENAMES
    )
    label = _french_date_label(selected)
    results_commit = _commit_and_push_exact_paths(
        project_directory=project,
        expected_paths=results_paths,
        expected_parent_commit=starting_commit,
        commit_message=f"Enregistrer le lot fantome MLB du {label}",
        stage=DailyPredictionStage.RESULTS_PUBLICATION,
    )

    try:
        certification = _certify_prediction(
            selected,
            results_commit,
            project_directory=project,
        )
    except DailyPredictionAutomationError:
        raise
    except Exception as error:
        raise DailyPredictionAutomationError(
            DailyPredictionStage.CERTIFICATION,
            "La certification s'est arretee en conservant ses preuves locales.",
        ) from error
    if (
        certification.batch_id != prediction.batch_id
        or certification.results_commit != results_commit
    ):
        raise DailyPredictionAutomationError(
            DailyPredictionStage.CERTIFICATION,
            "La certification ne correspond pas au lot qui vient d'etre publie.",
        )
    certification_paths = tuple(
        sorted(
            (
                certification.certification_path.relative_to(project).as_posix(),
                certification.evidence_path.relative_to(project).as_posix(),
            )
        )
    )
    expected_certification_paths = tuple(
        sorted(
            (
                (CERTIFICATION_ROOT / f"{selected.isoformat()}.json").as_posix(),
                (
                    CERTIFICATION_ROOT
                    / f"{selected.isoformat()}.remote.json.gz"
                ).as_posix(),
            )
        )
    )
    if certification_paths != expected_certification_paths:
        raise DailyPredictionAutomationError(
            DailyPredictionStage.CERTIFICATION,
            "Les chemins de certification produits sont inattendus.",
        )
    certification_commit = _commit_and_push_exact_paths(
        project_directory=project,
        expected_paths=certification_paths,
        expected_parent_commit=results_commit,
        commit_message=(
            f"Enregistrer la certification prospective MLB du {label}"
        ),
        stage=DailyPredictionStage.CERTIFICATION_PUBLICATION,
    )
    return DailyPredictionPublication(
        target_date=selected,
        batch_id=prediction.batch_id,
        results_commit=results_commit,
        certification_commit=certification_commit,
        results_paths=results_paths,
        certification_paths=certification_paths,
    )


__all__ = [
    "DailyAction",
    "DailyActionState",
    "DailyOperationsError",
    "DailyOperationsOverview",
    "DailyPredictionAutomationError",
    "DailyPredictionPublication",
    "DailyPredictionStage",
    "GitWorkspaceState",
    "LocalGameDayState",
    "PredictionSlotState",
    "build_daily_operations_overview",
    "execute_daily_prediction_publication",
    "inspect_daily_operations",
    "inspect_git_workspace",
    "inspect_prediction_slot",
    "load_local_game_day_state",
    "refresh_daily_mlb_data",
]
