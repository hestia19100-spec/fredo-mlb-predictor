"""Fondations locales et inertes de la prediction fantome MLB v2.

Ce module ne sait volontairement ni activer ni executer une prediction. Il
valide l'apercu statique, construit les formats canoniques et inspecte en
lecture seule un eventuel creneau deja consomme. Il n'importe aucun composant
d'ingestion, de base de donnees ou de modele.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import date, datetime
from enum import Enum
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tempfile
from typing import Any, Mapping, Sequence


PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
SHADOW_PROTOCOL_RELATIVE_PATH = PurePosixPath(
    "shadow_protocols/logistic_team_form_v1_platt_shadow_v2.json"
)
DEFAULT_SHADOW_PROTOCOL_PATH = PROJECT_DIRECTORY.joinpath(
    *SHADOW_PROTOCOL_RELATIVE_PATH.parts
)

EXPECTED_SHADOW_PROTOCOL_SHA256 = (
    "4dcae9e85bb9ed5b3f4a9f961491d872"
    "56965c42e97d11e70f31f02f52cc49c9"
)
EXPECTED_SHADOW_PROTOCOL_VERSION = 2
EXPECTED_SHADOW_PROTOCOL_STATUS = (
    "REGISTERED_BEFORE_FIRST_SHADOW_V2_PREDICTION"
)
EXPECTED_SHADOW_PROTOCOL_REGISTERED_ON = date(2026, 8, 31)
EXPECTED_TARGET_SEASON = 2026
EXPECTED_PREVIEW_MODE = "PREVIEW_WITHOUT_MODEL_OR_PREDICTIONS"

SHADOW_RESULT_ROOT_RELATIVE_PATH = PurePosixPath(
    "shadow_results/logistic_team_form_v1_platt_shadow_v2"
)
SHADOW_SLOT_SUCCESS_FILENAMES = (
    "RESERVED",
    "activation_reverification.remote.json.gz",
    "source_snapshot.json.gz",
    "candidate_ledger.csv",
    "features.csv",
    "predictions.csv",
    "receipt.json",
    "COMPLETED",
)

EXPECTED_SERVICE_MODULE_PATH = "src/shadow_prediction.py"
EXPECTED_MODEL_ARTIFACT_PATH = (
    "models/logistic_team_form_v1_platt.joblib"
)
EXPECTED_MODEL_ARTIFACT_SHA256 = (
    "e0d4d2421ba076072c7ef8b3bc97dd9a"
    "341e26c62828a0ad9ba43f30da15ff55"
)
EXPECTED_ARTIFACT_MANIFEST_PATH = (
    "model_artifacts/logistic_team_form_v1_platt.json"
)
EXPECTED_ARTIFACT_MANIFEST_SHA256 = (
    "a7375d5376baa043b3365cff713cb717"
    "010ad812efee472b594fa454306c96ae"
)
EXPECTED_MODEL_PROTOCOL_PATH = "model_protocols/logistic_team_form_v1.json"
EXPECTED_MODEL_PROTOCOL_SHA256 = (
    "c4cb1af750619967514d37ae3a5a47a6"
    "a04255aeaccb20c5e94533dc4d138451"
)

_CANONICAL_DATE_PATTERN = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}\Z")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_GIT_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}\Z")
_UTC_TIMESTAMP_PATTERN = re.compile(
    r"([0-9]{4})-([0-9]{2})-([0-9]{2})T"
    r"([0-9]{2}):([0-9]{2}):([0-9]{2})Z\Z"
)
_FIXED_6_PATTERN = re.compile(r"[0-9]+\.[0-9]{6}\Z")

_FEATURE_ROW_FIELD_NAMES = (
    "prediction_id",
    "batch_id",
    "game_id",
    "occurrence_key",
    "season",
    "official_date",
    "away_team_id",
    "home_team_id",
    "scheduled_start_utc",
    "feature_as_of_date",
    "away_max_source_date",
    "home_max_source_date",
    "away_games_before",
    "away_win_pct_before",
    "away_runs_scored_per_game_before",
    "away_runs_allowed_per_game_before",
    "home_games_before",
    "home_win_pct_before",
    "home_runs_scored_per_game_before",
    "home_runs_allowed_per_game_before",
)
_FEATURE_ROW_SHA256_INDEXES = (0, 1, 3)
_FEATURE_ROW_INTEGER_INDEXES = (2, 4, 6, 7, 12, 16)
_FEATURE_ROW_DATE_INDEXES = (5, 9, 10, 11)
_FEATURE_ROW_TIMESTAMP_INDEX = 8
_FEATURE_ROW_FIXED_6_INDEXES = (13, 14, 15, 17, 18, 19)

_RESERVED_MARKER_KEYS = frozenset(
    {
        "marker_schema_version",
        "batch_id",
        "slot_key",
        "target_official_date",
        "reserved_at_utc",
        "shadow_protocol_sha256",
        "execution_manifest_sha256",
        "runtime_code_commit",
    }
)
_FAILED_MARKER_KEYS = frozenset(
    {
        "marker_schema_version",
        "batch_id",
        "slot_key",
        "target_official_date",
        "failed_at_utc",
        "stage",
        "error_type",
        "error_message",
        "shadow_protocol_sha256",
        "execution_manifest_sha256",
        "runtime_code_commit",
    }
)
_COMPLETED_MARKER_KEYS = frozenset(
    {
        "marker_schema_version",
        "batch_id",
        "receipt_path",
        "receipt_sha256",
        "completed_at_utc",
    }
)
_RECEIPT_TOP_LEVEL_KEYS = frozenset(
    {
        "receipt_schema_version",
        "batch",
        "activation",
        "times",
        "schedule_http_response",
        "lineage",
        "source",
        "counts",
        "output_hashes",
        "model_invariants",
        "negative_attestations",
        "runtime_versions",
    }
)
_RECEIPT_SECTION_KEYS = {
    "batch": frozenset(
        {
            "batch_id",
            "slot_key",
            "target_official_date",
            "status",
            "earliest_predicted_scheduled_start_utc",
        }
    ),
    "activation": frozenset(
        {
            "execution_manifest_introduction_commit",
            "activation_introduction_commit",
            "activation_path",
            "activation_sha256",
            "activation_verified_at_utc",
            "activation_remote_reverified_at_utc",
            "activation_remote_ref",
            "activation_remote_reverification_query_url",
            "activation_remote_reverification_effective_url",
            "activation_remote_reverification_status_code",
            "activation_remote_reverification_redirect_count",
            "activation_remote_reverification_response_received_at_utc",
            "activation_remote_reverification_response_body_sha256",
            "activation_remote_reverification_evidence_path",
            "activation_remote_reverification_evidence_sha256",
            "minimum_target_official_date",
        }
    ),
    "times": frozenset(
        {
            "started_at_utc",
            "reserved_at_utc",
            "schedule_observed_at_utc",
            "information_cutoff_utc",
            "issued_at_utc",
            "receipt_finalized_at_utc",
            "mlb_http_date_utc",
            "mlb_http_response_received_at_utc",
            "clock_skew_seconds",
            "schedule_age_seconds",
        }
    ),
    "schedule_http_response": frozenset(
        {
            "effective_url",
            "status_code",
            "redirect_count",
            "date_header_raw",
            "date_header_utc",
            "received_at_utc",
            "body_sha256",
        }
    ),
    "lineage": frozenset(
        {
            "runtime_code_commit",
            "shadow_service_module_sha256",
            "shadow_protocol_sha256",
            "execution_manifest_sha256",
            "model_artifact_sha256",
            "artifact_manifest_sha256",
            "model_protocol_sha256",
            "evaluation_protocol_sha256",
            "evaluation_report_sha256",
            "evaluation_results_commit",
        }
    ),
    "source": frozenset(
        {
            "sqlite_snapshot_sha256",
            "sqlite_snapshot_size_bytes",
            "source_snapshot_path",
            "source_snapshot_sha256",
            "schedule_ingestion_run_id",
            "schedule_source",
            "schedule_requested_start_date",
            "schedule_requested_end_date",
            "schedule_game_types",
            "schedule_request_parameters_json",
            "schedule_ingestion_completed_at_utc",
            "schedule_raw_archive_path",
            "schedule_raw_archive_sha256",
        }
    ),
    "counts": frozenset(
        {
            "schedule_games",
            "eligible_games",
            "predicted_games",
            "excluded_games_by_reason",
        }
    ),
    "output_hashes": frozenset(
        {
            "activation_reverification_evidence_sha256",
            "candidate_ledger_sha256",
            "features_sha256",
            "predictions_sha256",
        }
    ),
    "model_invariants": frozenset(
        {
            "fit_calls",
            "partial_fit_calls",
            "recalibration_calls",
            "threshold_tuning_calls",
            "feature_selection_calls",
            "predict_proba_calls",
            "artifact_state_sha256_before",
            "artifact_state_sha256_after",
            "artifact_state_unchanged",
            "warning_policy_id",
            "approved_compatibility_warning_count",
            "unexpected_warning_count",
        }
    ),
    "negative_attestations": frozenset(
        {
            "ALL_TARGET_GAMES_UNSTARTED_AT_INFORMATION_CUTOFF",
            "TARGET_OUTCOMES_NOT_AVAILABLE_AT_INFORMATION_CUTOFF",
            "TARGET_SCORE_VALUES_REDACTED_FROM_TRACKED_TARGET_ROWS",
            "TARGET_SCORES_NOT_USED_AS_FEATURES",
            "ODDS_NOT_READ",
            "BETTING_RECOMMENDATIONS_NOT_COMPUTED",
            "SEASON_2026_METRICS_NOT_COMPUTED",
        }
    ),
    "runtime_versions": frozenset(
        {
            "python",
            "numpy",
            "pandas",
            "scipy",
            "scikit_learn",
            "joblib",
        }
    ),
}
_EXCLUDED_GAMES_BY_REASON_KEYS = frozenset(
    {
        "POSTPONED",
        "CANCELLED",
        "START_TIME_MISSING",
        "INSUFFICIENT_BOTH_HISTORY",
        "INSUFFICIENT_AWAY_HISTORY",
        "INSUFFICIENT_HOME_HISTORY",
    }
)
_BATCH_STATUS_DOMAIN = frozenset(
    {
        "COMPLETED_WITH_PREDICTIONS",
        "COMPLETED_NO_ELIGIBLE_GAMES",
    }
)


class ShadowPredictionError(RuntimeError):
    """Erreur qui interdit de produire meme un apercu fiable."""


class ShadowPublicationConflictError(ShadowPredictionError):
    """Refus attendu lorsqu'un chemin append-only est deja consomme."""


class ShadowPredictionSlotConsumedError(ShadowPublicationConflictError):
    """Refus definitif lorsqu'un slot journalier n'est plus absent."""


class ShadowPredictionSlotState(str, Enum):
    """Etats fermes d'un creneau journalier v2 deja inspecte."""

    ABSENT = "ABSENT"
    COMPLETED_EXACT = "COMPLETED_EXACT"
    COMPLETED_MISMATCH = "COMPLETED_MISMATCH"
    FAILED_CONSUMED = "FAILED_CONSUMED"
    INCOMPLETE_CONSUMED = "INCOMPLETE_CONSUMED"


@dataclass(frozen=True, slots=True)
class ShadowPredictionSlotInspection:
    """Resultat local d'inspection; seul un doublon exact rend son recu."""

    state: ShadowPredictionSlotState
    slot_path: Path
    slot_key: str
    batch_id: str
    receipt: dict[str, Any] | None


@dataclass(frozen=True, slots=True)
class ShadowPredictionSlotReservation:
    """Preuve locale retournee apres creation exclusive de RESERVED."""

    slot_path: Path
    slot_key: str
    batch_id: str
    reserved_marker: dict[str, Any]
    reserved_marker_sha256: str


@dataclass(frozen=True, slots=True)
class ShadowPredictionSlotFailure:
    """Preuve locale retournee apres publication exclusive de FAILED.json."""

    slot_path: Path
    failed_marker: dict[str, Any]
    failed_marker_sha256: str


def _validate_json_value(
    value: object,
    *,
    path: str = "$",
    ancestors: set[int] | None = None,
) -> None:
    """Refuse les extensions Python ambigues avant canonicalisation JSON."""
    value_type = type(value)
    if value is None or value_type in (bool, int, str):
        return
    if value_type is float:
        if not math.isfinite(value):
            raise ShadowPredictionError(
                f"Valeur JSON non finie interdite a {path}."
            )
        return

    if value_type not in (list, dict):
        raise ShadowPredictionError(
            f"Type non JSON interdit a {path} : {value_type.__name__}."
        )

    current_id = id(value)
    active_ancestors = ancestors if ancestors is not None else set()
    if current_id in active_ancestors:
        raise ShadowPredictionError(
            f"Reference circulaire interdite a {path}."
        )
    active_ancestors.add(current_id)
    try:
        if value_type is list:
            for index, item in enumerate(value):
                _validate_json_value(
                    item,
                    path=f"{path}[{index}]",
                    ancestors=active_ancestors,
                )
            return

        for key, item in value.items():
            if type(key) is not str:
                raise ShadowPredictionError(
                    f"Cle JSON non textuelle interdite a {path}."
                )
            _validate_json_value(
                item,
                path=f"{path}.{key}",
                ancestors=active_ancestors,
            )
    finally:
        active_ancestors.remove(current_id)


def _canonical_json_bytes(value: object) -> bytes:
    """Applique exactement la canonicalisation de hachage du protocole v2."""
    _validate_json_value(value)
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise ShadowPredictionError(
            "Valeur impossible a canonicaliser en JSON UTF-8."
        ) from error


def _canonical_json_file_bytes(value: object) -> bytes:
    """Ajoute l'unique LF terminal impose aux fichiers JSON v2."""
    return _canonical_json_bytes(value) + b"\n"


def _csv_field_text(value: object, *, row_index: int, column: str) -> str:
    """Convertit uniquement les scalaires dont le rendu CSV est explicite."""
    if value is None:
        return ""
    if type(value) is str:
        text = value
    elif type(value) is int:
        text = str(value)
    else:
        raise ShadowPredictionError(
            "Valeur CSV non canonique a la ligne "
            f"{row_index}, colonne {column!r} : "
            f"{type(value).__name__}."
        )
    if "\r" in text:
        raise ShadowPredictionError(
            "Retour chariot interdit dans une valeur CSV canonique a la "
            f"ligne {row_index}, colonne {column!r}."
        )
    return text


def _canonical_csv_bytes(
    columns: Sequence[str],
    rows: Sequence[Sequence[object]],
) -> bytes:
    """Serialise un tableau deja ordonne en CSV UTF-8/LF a largeur fixe."""
    if type(columns) not in (list, tuple) or not columns:
        raise ShadowPredictionError(
            "Les colonnes CSV doivent etre une liste ordonnee non vide."
        )
    canonical_columns: list[str] = []
    for index, column in enumerate(columns):
        if type(column) is not str or not column or "\r" in column:
            raise ShadowPredictionError(
                f"Nom de colonne CSV invalide a l'index {index}."
            )
        canonical_columns.append(column)
    if len(set(canonical_columns)) != len(canonical_columns):
        raise ShadowPredictionError(
            "Les noms de colonnes CSV doivent etre uniques."
        )

    if type(rows) not in (list, tuple):
        raise ShadowPredictionError(
            "Les lignes CSV doivent etre une liste ordonnee."
        )

    canonical_rows: list[list[str]] = []
    for row_index, row in enumerate(rows, start=1):
        if type(row) not in (list, tuple):
            raise ShadowPredictionError(
                f"La ligne CSV {row_index} doit etre une sequence ordonnee."
            )
        if len(row) != len(canonical_columns):
            raise ShadowPredictionError(
                f"Largeur CSV invalide a la ligne {row_index} : "
                f"{len(row)}, attendu {len(canonical_columns)}."
            )
        canonical_rows.append(
            [
                _csv_field_text(
                    value,
                    row_index=row_index,
                    column=canonical_columns[column_index],
                )
                for column_index, value in enumerate(row)
            ]
        )

    destination = io.StringIO(newline="")
    writer = csv.writer(
        destination,
        delimiter=",",
        quotechar='"',
        quoting=csv.QUOTE_MINIMAL,
        lineterminator="\n",
    )
    try:
        writer.writerow(canonical_columns)
        writer.writerows(canonical_rows)
        return destination.getvalue().encode("utf-8")
    except (csv.Error, UnicodeEncodeError) as error:
        raise ShadowPredictionError(
            "Valeur impossible a serialiser en CSV UTF-8 canonique."
        ) from error


def _canonical_gzip_bytes(payload: bytes) -> bytes:
    """Compresse des octets en memoire avec l'en-tete gzip fige v2."""
    if type(payload) is not bytes:
        raise ShadowPredictionError(
            "Le contenu gzip canonique doit etre fourni en octets exacts."
        )
    destination = io.BytesIO()
    with gzip.GzipFile(
        filename="",
        mode="wb",
        compresslevel=9,
        fileobj=destination,
        mtime=0,
    ) as gzip_file:
        gzip_file.write(payload)
    return destination.getvalue()


def _fsync_parent_directory(directory: Path) -> None:
    """Synchronise l'entree de repertoire sur les plateformes POSIX."""
    if os.name == "nt":
        return

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    directory_descriptor = os.open(directory, flags)
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)


def _publish_exclusive_verified(destination: Path, content: bytes) -> str:
    """Publie une fois, synchronise, relit et renvoie le SHA-256 exact.

    Le parent doit deja exister. Un lien physique cree la destination sans
    aucune possibilite de remplacement, meme si deux processus publient au
    meme instant. Une destination publiee n'est jamais supprimee par ce
    helper, y compris lorsqu'une verification ulterieure echoue.
    """
    if not isinstance(destination, Path):
        raise ShadowPredictionError(
            "La destination append-only doit etre un objet Path."
        )
    if type(content) is not bytes:
        raise ShadowPredictionError(
            "Le contenu append-only doit etre fourni en octets exacts."
        )
    if not destination.name:
        raise ShadowPredictionError(
            "La destination append-only doit nommer un fichier."
        )

    parent = destination.parent
    parent_mode = _lstat_mode(parent)
    if not isinstance(parent_mode, int) or not stat.S_ISDIR(parent_mode):
        raise ShadowPredictionError(
            "Le repertoire parent doit exister, etre local et ne pas etre "
            "symbolique avant toute publication."
        )

    expected_sha256 = hashlib.sha256(content).hexdigest()
    temporary_path: Path | None = None
    published = False

    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=parent,
        )
        temporary_path = Path(temporary_name)

        try:
            with os.fdopen(descriptor, "wb") as temporary_file:
                temporary_file.write(content)
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
        except BaseException:
            try:
                os.close(descriptor)
            except OSError:
                pass
            raise

        try:
            os.link(temporary_path, destination)
        except FileExistsError as error:
            raise ShadowPublicationConflictError(
                f"La destination append-only existe deja : {destination}."
            ) from error

        published = True
        temporary_path.unlink()
        temporary_path = None
        _fsync_parent_directory(parent)

        persisted = destination.read_bytes()
        persisted_sha256 = hashlib.sha256(persisted).hexdigest()
        if persisted != content or persisted_sha256 != expected_sha256:
            raise ShadowPredictionError(
                "La relecture append-only ne correspond pas aux octets "
                "publies."
            )
        return expected_sha256
    except (ShadowPredictionError, OSError):
        raise
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                if not published:
                    raise


def _format_feature_rate(value: object) -> str:
    """Formate un taux fini non negatif avec exactement six decimales."""
    if type(value) not in (int, float):
        raise ShadowPredictionError(
            "Un taux de variable doit etre un nombre JSON, hors booleen."
        )
    if type(value) is float and not math.isfinite(value):
        raise ShadowPredictionError(
            "Un taux de variable doit etre fini."
        )
    if value < 0:
        raise ShadowPredictionError(
            "Un taux de variable doit etre non negatif."
        )
    rendered = format(value, ".6f")
    if not _FIXED_6_PATTERN.fullmatch(rendered):
        raise ShadowPredictionError(
            "Un taux de variable doit avoir un rendu decimal non negatif."
        )
    return rendered


def _format_probability_float(value: object) -> str:
    """Formate une probabilite finie de [0, 1] avec la regle .17g."""
    if type(value) not in (int, float):
        raise ShadowPredictionError(
            "Une probabilite doit etre un nombre JSON, hors booleen."
        )
    if type(value) is float and not math.isfinite(value):
        raise ShadowPredictionError("Une probabilite doit etre finie.")
    if value < 0 or value > 1:
        raise ShadowPredictionError(
            "Une probabilite doit appartenir a l'intervalle [0, 1]."
        )
    rendered = format(value, ".17g")
    if rendered.startswith("-"):
        raise ShadowPredictionError(
            "Une probabilite ne peut pas utiliser un zero negatif."
        )
    return rendered


def _sha256_identifier(preimage: object) -> str:
    """Hache une preimage JSON-array sans saut de ligne terminal."""
    if type(preimage) is not list or not preimage:
        raise ShadowPredictionError(
            "La preimage d'un identifiant doit etre un tableau JSON non vide."
        )
    if type(preimage[0]) is not str:
        raise ShadowPredictionError(
            "Le domaine d'un identifiant doit etre une chaine JSON."
        )
    return hashlib.sha256(_canonical_json_bytes(preimage)).hexdigest()


def _require_sha256(value: object, *, field: str) -> str:
    if type(value) is not str or not _SHA256_PATTERN.fullmatch(value):
        raise ShadowPredictionError(
            f"{field} doit contenir exactement 64 caracteres hexadecimaux "
            "ASCII minuscules."
        )
    return value


def _require_git_commit(value: object, *, field: str) -> str:
    if type(value) is not str or not _GIT_COMMIT_PATTERN.fullmatch(value):
        raise ShadowPredictionError(
            f"{field} doit contenir exactement 40 caracteres hexadecimaux "
            "ASCII minuscules."
        )
    return value


def _require_nonempty_text(value: object, *, field: str) -> str:
    if type(value) is not str or not value.strip():
        raise ShadowPredictionError(
            f"{field} doit etre une chaine non vide."
        )
    return value


def _require_integer(value: object, *, field: str) -> int:
    if type(value) is not int:
        raise ShadowPredictionError(
            f"{field} doit etre un entier JSON; les booleens sont interdits."
        )
    return value


def _require_date_string(value: object, *, field: str) -> str:
    if type(value) is not str or not _CANONICAL_DATE_PATTERN.fullmatch(
        value
    ):
        raise ShadowPredictionError(
            f"{field} doit respecter exactement YYYY-MM-DD."
        )
    try:
        parsed = date.fromisoformat(value)
    except ValueError as error:
        raise ShadowPredictionError(
            f"{field} n'est pas une date calendaire valide : {value!r}."
        ) from error
    if parsed.isoformat() != value:
        raise ShadowPredictionError(
            f"{field} doit etre une date ISO canonique."
        )
    return value


def _require_utc_timestamp(value: object, *, field: str) -> str:
    if type(value) is not str:
        raise ShadowPredictionError(
            f"{field} doit etre une chaine RFC3339_SECONDS_Z."
        )
    match = _UTC_TIMESTAMP_PATTERN.fullmatch(value)
    if match is None:
        raise ShadowPredictionError(
            f"{field} doit respecter exactement RFC3339_SECONDS_Z."
        )
    try:
        datetime(*(int(component) for component in match.groups()))
    except ValueError as error:
        raise ShadowPredictionError(
            f"{field} n'est pas un instant UTC valide : {value!r}."
        ) from error
    return value


def _require_fixed_6_string(value: object, *, field: str) -> str:
    if type(value) is not str or not _FIXED_6_PATTERN.fullmatch(value):
        raise ShadowPredictionError(
            f"{field} doit etre une chaine decimale non negative avec "
            "exactement six decimales."
        )
    return value


def build_slot_key(
    *,
    shadow_protocol_sha256: str,
    target_official_date: str,
) -> str:
    """Construit l'identifiant v2 d'un slot sans aucune entree-sortie."""
    protocol_hash = _require_sha256(
        shadow_protocol_sha256,
        field="shadow_protocol_sha256",
    )
    target_date = _require_date_string(
        target_official_date,
        field="target_official_date",
    )
    return _sha256_identifier(
        ["shadow_slot_v2", protocol_hash, target_date]
    )


def build_batch_id(
    *,
    slot_key: str,
    execution_manifest_sha256: str,
    model_artifact_sha256: str,
) -> str:
    """Construit l'identifiant v2 d'un lot sans consulter ses fichiers."""
    validated_slot_key = _require_sha256(slot_key, field="slot_key")
    manifest_hash = _require_sha256(
        execution_manifest_sha256,
        field="execution_manifest_sha256",
    )
    model_hash = _require_sha256(
        model_artifact_sha256,
        field="model_artifact_sha256",
    )
    return _sha256_identifier(
        [
            "shadow_batch_v2",
            validated_slot_key,
            manifest_hash,
            model_hash,
        ]
    )


def build_occurrence_key(
    *,
    game_id: int,
    official_date_at_snapshot: str,
    scheduled_start_utc_at_snapshot_or_null: str | None,
) -> str:
    """Construit une occurrence v2; seul ce hachage accepte un debut nul."""
    validated_game_id = _require_integer(game_id, field="game_id")
    official_date = _require_date_string(
        official_date_at_snapshot,
        field="official_date_at_snapshot",
    )
    scheduled_start: str | None
    if scheduled_start_utc_at_snapshot_or_null is None:
        scheduled_start = None
    else:
        scheduled_start = _require_utc_timestamp(
            scheduled_start_utc_at_snapshot_or_null,
            field="scheduled_start_utc_at_snapshot_or_null",
        )
    return _sha256_identifier(
        [
            "shadow_occurrence_v2",
            validated_game_id,
            official_date,
            scheduled_start,
        ]
    )


def build_prediction_id(
    *,
    shadow_protocol_sha256: str,
    game_id: int,
    official_date_at_snapshot: str,
    scheduled_start_utc_at_snapshot: str,
) -> str:
    """Construit l'identifiant v2 d'une prediction admissible."""
    protocol_hash = _require_sha256(
        shadow_protocol_sha256,
        field="shadow_protocol_sha256",
    )
    validated_game_id = _require_integer(game_id, field="game_id")
    official_date = _require_date_string(
        official_date_at_snapshot,
        field="official_date_at_snapshot",
    )
    scheduled_start = _require_utc_timestamp(
        scheduled_start_utc_at_snapshot,
        field="scheduled_start_utc_at_snapshot",
    )
    return _sha256_identifier(
        [
            "shadow_prediction_v2",
            protocol_hash,
            validated_game_id,
            official_date,
            scheduled_start,
        ]
    )


def build_feature_row_sha256(
    *,
    feature_row_values_in_features_columns_exact_order_excluding_feature_row_sha256:
        list[object],
) -> str:
    """Hache les 20 valeurs v2 d'une ligne de variables, dans l'ordre fige."""
    source_values = (
        feature_row_values_in_features_columns_exact_order_excluding_feature_row_sha256
    )
    if type(source_values) is not list:
        raise ShadowPredictionError(
            "Les valeurs de feature row doivent etre un tableau JSON."
        )
    if len(source_values) != len(_FEATURE_ROW_FIELD_NAMES):
        raise ShadowPredictionError(
            "Une feature row v2 doit contenir exactement "
            f"{len(_FEATURE_ROW_FIELD_NAMES)} valeurs."
        )

    values = list(source_values)
    for index in _FEATURE_ROW_SHA256_INDEXES:
        _require_sha256(values[index], field=_FEATURE_ROW_FIELD_NAMES[index])
    for index in _FEATURE_ROW_INTEGER_INDEXES:
        _require_integer(values[index], field=_FEATURE_ROW_FIELD_NAMES[index])
    for index in _FEATURE_ROW_DATE_INDEXES:
        _require_date_string(
            values[index],
            field=_FEATURE_ROW_FIELD_NAMES[index],
        )
    _require_utc_timestamp(
        values[_FEATURE_ROW_TIMESTAMP_INDEX],
        field=_FEATURE_ROW_FIELD_NAMES[_FEATURE_ROW_TIMESTAMP_INDEX],
    )
    for index in _FEATURE_ROW_FIXED_6_INDEXES:
        _require_fixed_6_string(
            values[index],
            field=_FEATURE_ROW_FIELD_NAMES[index],
        )

    return _sha256_identifier(["shadow_feature_row_v2", values])


@dataclass(frozen=True, slots=True)
class ShadowPredictionPreview:
    """Resultat deterministe d'un controle statique, sans execution."""

    mode: str
    shadow_protocol_version: int
    shadow_protocol_path: str
    shadow_protocol_sha256: str
    shadow_protocol_registered_on: str
    shadow_protocol_status: str
    target_official_date: str
    target_season: int
    target_validation_scope: str
    execution_ready: bool
    execution_manifest_read: bool
    activation_read: bool
    model_artifact_read: bool
    model_deserialized: bool
    sqlite_read: bool
    network_request_performed: bool
    output_slot_reserved: bool
    output_files_created: bool
    predictions_computed: bool

    def to_dict(self) -> dict[str, object]:
        """Retourne une representation JSON sans etat ni horodatage."""
        return asdict(self)

    def to_canonical_json(self) -> str:
        """Serialise l'apercu avec un encodage stable entre deux appels."""
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )


def _reject_duplicate_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    """Refuse une ambiguite JSON au lieu de conserver la derniere cle."""
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ShadowPredictionError(
                f"Cle JSON dupliquee dans le protocole v2 : {key!r}."
            )
        result[key] = value
    return result


def _require_mapping(
    value: object,
    *,
    description: str,
) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ShadowPredictionError(
            f"Objet JSON attendu pour {description}."
        )
    return value


def _require_exact(
    mapping: Mapping[str, Any],
    key: str,
    expected: object,
    *,
    context: str,
) -> None:
    actual = mapping.get(key)
    if type(actual) is not type(expected) or actual != expected:
        raise ShadowPredictionError(
            f"Valeur v2 invalide pour {context}.{key} : "
            f"{actual!r}, attendu {expected!r}."
        )


_PATH_MISSING = object()
_PATH_UNREADABLE = object()


def _lstat_mode(path: Path) -> int | object:
    """Retourne le mode sans accepter lien ou reparse point Windows."""
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return _PATH_MISSING
    except (OSError, ValueError):
        return _PATH_UNREADABLE
    reparse_attribute = getattr(
        stat,
        "FILE_ATTRIBUTE_REPARSE_POINT",
        0x0400,
    )
    if getattr(metadata, "st_file_attributes", 0) & reparse_attribute:
        return _PATH_UNREADABLE
    return metadata.st_mode


def _has_exact_keys(
    value: object,
    expected_keys: frozenset[str],
) -> bool:
    return type(value) is dict and frozenset(value) == expected_keys


def _read_canonical_json_object(
    path: Path,
) -> tuple[dict[str, Any], bytes] | None:
    """Lit un fichier JSON regulier, canonique et sans cle dupliquee."""
    mode = _lstat_mode(path)
    if not isinstance(mode, int) or not stat.S_ISREG(mode):
        return None
    try:
        content = path.read_bytes()
        payload = json.loads(
            content.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
        if type(payload) is not dict:
            return None
        if _canonical_json_file_bytes(payload) != content:
            return None
    except (
        OSError,
        UnicodeError,
        ValueError,
        ShadowPredictionError,
        RecursionError,
    ):
        return None
    return payload, content


def _valid_reserved_marker(
    marker: Mapping[str, Any],
    *,
    batch_id: str,
    slot_key: str,
    target_official_date: str,
    shadow_protocol_sha256: str,
    execution_manifest_sha256: str,
) -> bool:
    if not _has_exact_keys(marker, _RESERVED_MARKER_KEYS):
        return False
    expected_values = {
        "marker_schema_version": 1,
        "batch_id": batch_id,
        "slot_key": slot_key,
        "target_official_date": target_official_date,
        "shadow_protocol_sha256": shadow_protocol_sha256,
        "execution_manifest_sha256": execution_manifest_sha256,
    }
    if any(
        type(marker.get(key)) is not type(expected)
        or marker.get(key) != expected
        for key, expected in expected_values.items()
    ):
        return False
    runtime_code_commit = marker.get("runtime_code_commit")
    if (
        type(runtime_code_commit) is not str
        or not _GIT_COMMIT_PATTERN.fullmatch(runtime_code_commit)
    ):
        return False
    try:
        _require_utc_timestamp(
            marker.get("reserved_at_utc"),
            field="RESERVED.reserved_at_utc",
        )
    except ShadowPredictionError:
        return False
    return True


def _valid_receipt(
    receipt: Mapping[str, Any],
    *,
    reserved: Mapping[str, Any],
    batch_id: str,
    slot_key: str,
    target_official_date: str,
    shadow_protocol_sha256: str,
    execution_manifest_sha256: str,
    model_artifact_sha256: str,
) -> bool:
    if not _has_exact_keys(receipt, _RECEIPT_TOP_LEVEL_KEYS):
        return False
    if type(receipt.get("receipt_schema_version")) is not int:
        return False
    if receipt.get("receipt_schema_version") != 1:
        return False
    for section_name, expected_keys in _RECEIPT_SECTION_KEYS.items():
        if not _has_exact_keys(receipt.get(section_name), expected_keys):
            return False

    batch = receipt["batch"]
    lineage = receipt["lineage"]
    times = receipt["times"]
    counts = receipt["counts"]
    output_hashes = receipt["output_hashes"]
    source = receipt["source"]
    negative_attestations = receipt["negative_attestations"]
    assert isinstance(batch, dict)
    assert isinstance(lineage, dict)
    assert isinstance(times, dict)
    assert isinstance(counts, dict)
    assert isinstance(output_hashes, dict)
    assert isinstance(source, dict)
    assert isinstance(negative_attestations, dict)

    exact_identity_values = {
        ("batch", "batch_id"): batch_id,
        ("batch", "slot_key"): slot_key,
        ("batch", "target_official_date"): target_official_date,
        (
            "lineage",
            "shadow_protocol_sha256",
        ): shadow_protocol_sha256,
        (
            "lineage",
            "execution_manifest_sha256",
        ): execution_manifest_sha256,
        ("lineage", "model_artifact_sha256"): model_artifact_sha256,
    }
    sections = {"batch": batch, "lineage": lineage}
    for (section_name, field), expected in exact_identity_values.items():
        actual = sections[section_name].get(field)
        if type(actual) is not type(expected) or actual != expected:
            return False

    runtime_code_commit = lineage.get("runtime_code_commit")
    if (
        type(runtime_code_commit) is not str
        or not _GIT_COMMIT_PATTERN.fullmatch(runtime_code_commit)
        or runtime_code_commit != reserved.get("runtime_code_commit")
    ):
        return False
    if times.get("reserved_at_utc") != reserved.get("reserved_at_utc"):
        return False

    status = batch.get("status")
    if type(status) is not str or status not in _BATCH_STATUS_DOMAIN:
        return False
    excluded = counts.get("excluded_games_by_reason")
    if not _has_exact_keys(excluded, _EXCLUDED_GAMES_BY_REASON_KEYS):
        return False
    assert isinstance(excluded, dict)
    count_values = [
        counts.get("schedule_games"),
        counts.get("eligible_games"),
        counts.get("predicted_games"),
        *excluded.values(),
    ]
    if any(type(value) is not int or value < 0 for value in count_values):
        return False
    if counts["predicted_games"] != counts["eligible_games"]:
        return False
    if counts["schedule_games"] != (
        counts["eligible_games"] + sum(excluded.values())
    ):
        return False
    predicted_games = counts["predicted_games"]
    expected_status = (
        "COMPLETED_WITH_PREDICTIONS"
        if predicted_games > 0
        else "COMPLETED_NO_ELIGIBLE_GAMES"
    )
    if status != expected_status:
        return False
    earliest_start = batch.get("earliest_predicted_scheduled_start_utc")
    if predicted_games == 0:
        if earliest_start is not None:
            return False
    else:
        try:
            _require_utc_timestamp(
                earliest_start,
                field=(
                    "receipt.batch."
                    "earliest_predicted_scheduled_start_utc"
                ),
            )
        except ShadowPredictionError:
            return False

    for field in _RECEIPT_SECTION_KEYS["output_hashes"]:
        try:
            _require_sha256(
                output_hashes.get(field),
                field=f"receipt.output_hashes.{field}",
            )
        except ShadowPredictionError:
            return False
    for field in (
        "sqlite_snapshot_sha256",
        "source_snapshot_sha256",
        "schedule_raw_archive_sha256",
    ):
        try:
            _require_sha256(
                source.get(field),
                field=f"receipt.source.{field}",
            )
        except ShadowPredictionError:
            return False
    if any(value is not True for value in negative_attestations.values()):
        return False
    try:
        _require_utc_timestamp(
            times.get("receipt_finalized_at_utc"),
            field="receipt.times.receipt_finalized_at_utc",
        )
    except ShadowPredictionError:
        return False
    return True


def _valid_completed_marker(
    marker: Mapping[str, Any],
    *,
    receipt: Mapping[str, Any],
    receipt_bytes: bytes,
    receipt_path: str,
    batch_id: str,
) -> bool:
    if not _has_exact_keys(marker, _COMPLETED_MARKER_KEYS):
        return False
    expected_values = {
        "marker_schema_version": 1,
        "batch_id": batch_id,
        "receipt_path": receipt_path,
        "receipt_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
    }
    if any(
        type(marker.get(key)) is not type(expected)
        or marker.get(key) != expected
        for key, expected in expected_values.items()
    ):
        return False
    receipt_batch = receipt.get("batch")
    receipt_times = receipt.get("times")
    if not isinstance(receipt_batch, dict) or not isinstance(
        receipt_times,
        dict,
    ):
        return False
    if marker.get("batch_id") != receipt_batch.get("batch_id"):
        return False
    try:
        completed_at = _require_utc_timestamp(
            marker.get("completed_at_utc"),
            field="COMPLETED.completed_at_utc",
        )
        receipt_finalized_at = _require_utc_timestamp(
            receipt_times.get("receipt_finalized_at_utc"),
            field="receipt.times.receipt_finalized_at_utc",
        )
    except ShadowPredictionError:
        return False
    return receipt_finalized_at <= completed_at


def inspect_shadow_prediction_slot(
    target_official_date: date | str,
    *,
    shadow_protocol_sha256: str,
    execution_manifest_sha256: str,
    model_artifact_sha256: str,
    project_directory: Path = PROJECT_DIRECTORY,
) -> ShadowPredictionSlotInspection:
    """Inspecte un slot v2 local sans reserver, ecrire ni lire ses donnees.

    Les identifiants attendus sont derives ici a partir des valeurs runtime.
    Les CSV et gzip ne sont jamais ouverts : leur presence comme fichiers
    reguliers suffit a cette inspection d'idempotence.
    """
    target_date = _parse_target_official_date(
        target_official_date
    ).isoformat()
    protocol_hash = _require_sha256(
        shadow_protocol_sha256,
        field="shadow_protocol_sha256",
    )
    manifest_hash = _require_sha256(
        execution_manifest_sha256,
        field="execution_manifest_sha256",
    )
    model_hash = _require_sha256(
        model_artifact_sha256,
        field="model_artifact_sha256",
    )
    slot_key = build_slot_key(
        shadow_protocol_sha256=protocol_hash,
        target_official_date=target_date,
    )
    batch_id = build_batch_id(
        slot_key=slot_key,
        execution_manifest_sha256=manifest_hash,
        model_artifact_sha256=model_hash,
    )

    try:
        project_input = Path(project_directory).expanduser()
        project = Path(os.path.abspath(os.fspath(project_input)))
    except (TypeError, OSError, RuntimeError) as error:
        raise ShadowPredictionError(
            "Dossier du projet invalide pour l'inspection du slot."
        ) from error
    project_mode = _lstat_mode(project)
    if not isinstance(project_mode, int) or not stat.S_ISDIR(project_mode):
        raise ShadowPredictionError(
            "Le dossier du projet doit etre un repertoire local non "
            "symbolique."
        )

    slot_path = project.joinpath(
        *SHADOW_RESULT_ROOT_RELATIVE_PATH.parts,
        target_date,
    )

    def inspection(
        state: ShadowPredictionSlotState,
        receipt: dict[str, Any] | None = None,
    ) -> ShadowPredictionSlotInspection:
        return ShadowPredictionSlotInspection(
            state=state,
            slot_path=slot_path,
            slot_key=slot_key,
            batch_id=batch_id,
            receipt=receipt,
        )

    current = project
    for component in SHADOW_RESULT_ROOT_RELATIVE_PATH.parts:
        current = current / component
        mode = _lstat_mode(current)
        if mode is _PATH_MISSING:
            return inspection(ShadowPredictionSlotState.ABSENT)
        if not isinstance(mode, int) or not stat.S_ISDIR(mode):
            return inspection(
                ShadowPredictionSlotState.INCOMPLETE_CONSUMED
            )

    slot_mode = _lstat_mode(slot_path)
    if slot_mode is _PATH_MISSING:
        return inspection(ShadowPredictionSlotState.ABSENT)
    if not isinstance(slot_mode, int) or not stat.S_ISDIR(slot_mode):
        return inspection(ShadowPredictionSlotState.INCOMPLETE_CONSUMED)

    completed_path = slot_path / "COMPLETED"
    completed_mode = _lstat_mode(completed_path)
    if completed_mode is _PATH_MISSING:
        failed_mode = _lstat_mode(slot_path / "FAILED.json")
        if isinstance(failed_mode, int) and stat.S_ISREG(failed_mode):
            return inspection(ShadowPredictionSlotState.FAILED_CONSUMED)
        return inspection(ShadowPredictionSlotState.INCOMPLETE_CONSUMED)

    mismatch = ShadowPredictionSlotState.COMPLETED_MISMATCH
    if not isinstance(completed_mode, int) or not stat.S_ISREG(
        completed_mode
    ):
        return inspection(mismatch)

    try:
        entries = list(slot_path.iterdir())
    except OSError:
        return inspection(mismatch)
    expected_names = frozenset(SHADOW_SLOT_SUCCESS_FILENAMES)
    if len(entries) != len(expected_names):
        return inspection(mismatch)
    if frozenset(entry.name for entry in entries) != expected_names:
        return inspection(mismatch)
    if any(
        not isinstance(mode := _lstat_mode(entry), int)
        or not stat.S_ISREG(mode)
        for entry in entries
    ):
        return inspection(mismatch)

    reserved_result = _read_canonical_json_object(slot_path / "RESERVED")
    receipt_result = _read_canonical_json_object(slot_path / "receipt.json")
    completed_result = _read_canonical_json_object(completed_path)
    if (
        reserved_result is None
        or receipt_result is None
        or completed_result is None
    ):
        return inspection(mismatch)
    reserved, _reserved_bytes = reserved_result
    receipt, receipt_bytes = receipt_result
    completed, _completed_bytes = completed_result
    if not _valid_reserved_marker(
        reserved,
        batch_id=batch_id,
        slot_key=slot_key,
        target_official_date=target_date,
        shadow_protocol_sha256=protocol_hash,
        execution_manifest_sha256=manifest_hash,
    ):
        return inspection(mismatch)
    if not _valid_receipt(
        receipt,
        reserved=reserved,
        batch_id=batch_id,
        slot_key=slot_key,
        target_official_date=target_date,
        shadow_protocol_sha256=protocol_hash,
        execution_manifest_sha256=manifest_hash,
        model_artifact_sha256=model_hash,
    ):
        return inspection(mismatch)
    expected_receipt_path = (
        SHADOW_RESULT_ROOT_RELATIVE_PATH
        / target_date
        / "receipt.json"
    ).as_posix()
    if not _valid_completed_marker(
        completed,
        receipt=receipt,
        receipt_bytes=receipt_bytes,
        receipt_path=expected_receipt_path,
        batch_id=batch_id,
    ):
        return inspection(mismatch)
    return inspection(ShadowPredictionSlotState.COMPLETED_EXACT, receipt)


def _require_shadow_result_root(
    *,
    project_directory: Path,
) -> Path:
    """Exige la racine versionnee sans creer ni suivre aucun composant."""
    current = project_directory
    for component in SHADOW_RESULT_ROOT_RELATIVE_PATH.parts:
        current = current / component
        mode = _lstat_mode(current)
        if not isinstance(mode, int) or not stat.S_ISDIR(mode):
            raise ShadowPredictionError(
                "La racine versionnee des resultats fantomes doit deja "
                "exister et ne contenir aucun lien symbolique."
            )
    return current


def reserve_shadow_prediction_slot(
    target_official_date: date | str,
    *,
    reserved_at_utc: str,
    runtime_code_commit: str,
    shadow_protocol_sha256: str,
    execution_manifest_sha256: str,
    model_artifact_sha256: str,
    project_directory: Path = PROJECT_DIRECTORY,
) -> ShadowPredictionSlotReservation:
    """Reserve une fois un slot v2 absent et publie son premier fichier.

    Tous les instants et identifiants variables sont fournis par l'appelant :
    cette primitive ne consulte ni horloge, ni Git, ni reseau, ni SQLite, ni
    modele. Une fois le repertoire du slot cree, aucune erreur ne provoque sa
    suppression ou sa reutilisation.
    """
    target = _parse_target_official_date(target_official_date)
    target_date = target.isoformat()
    if target.year != EXPECTED_TARGET_SEASON:
        raise ShadowPredictionError(
            "Saison cible invalide pour une reservation fantome v2 : "
            f"{target.year}, attendu {EXPECTED_TARGET_SEASON}."
        )
    if target < EXPECTED_SHADOW_PROTOCOL_REGISTERED_ON:
        raise ShadowPredictionError(
            "La date cible ne peut pas preceder l'enregistrement du "
            "protocole fantome v2."
        )

    reserved_at = _require_utc_timestamp(
        reserved_at_utc,
        field="reserved_at_utc",
    )
    if target_date < reserved_at[:10]:
        raise ShadowPredictionError(
            "target_official_date ne peut pas preceder la date UTC de "
            "reserved_at_utc."
        )
    runtime_commit = _require_git_commit(
        runtime_code_commit,
        field="runtime_code_commit",
    )
    protocol_hash = _require_sha256(
        shadow_protocol_sha256,
        field="shadow_protocol_sha256",
    )
    if protocol_hash != EXPECTED_SHADOW_PROTOCOL_SHA256:
        raise ShadowPredictionError(
            "La reservation exige l'empreinte du protocole fantome v2 "
            "fige."
        )
    manifest_hash = _require_sha256(
        execution_manifest_sha256,
        field="execution_manifest_sha256",
    )
    model_hash = _require_sha256(
        model_artifact_sha256,
        field="model_artifact_sha256",
    )
    if model_hash != EXPECTED_MODEL_ARTIFACT_SHA256:
        raise ShadowPredictionError(
            "La reservation exige l'empreinte du modele valide et fige."
        )

    # L'inspection est volontairement la premiere operation sur l'arbre de
    # resultats. Elle derive aussi les identifiants attendus du slot et du lot.
    inspected = inspect_shadow_prediction_slot(
        target_date,
        shadow_protocol_sha256=protocol_hash,
        execution_manifest_sha256=manifest_hash,
        model_artifact_sha256=model_hash,
        project_directory=project_directory,
    )
    if inspected.state is not ShadowPredictionSlotState.ABSENT:
        raise ShadowPredictionSlotConsumedError(
            "Le slot fantome est deja consomme et ne peut jamais etre "
            f"reserve de nouveau : {inspected.state.value}."
        )

    reserved_marker: dict[str, Any] = {
        "marker_schema_version": 1,
        "batch_id": inspected.batch_id,
        "slot_key": inspected.slot_key,
        "target_official_date": target_date,
        "reserved_at_utc": reserved_at,
        "shadow_protocol_sha256": protocol_hash,
        "execution_manifest_sha256": manifest_hash,
        "runtime_code_commit": runtime_commit,
    }
    reserved_bytes = _canonical_json_file_bytes(reserved_marker)

    result_root = _require_shadow_result_root(
        project_directory=inspected.slot_path.parents[
            len(SHADOW_RESULT_ROOT_RELATIVE_PATH.parts)
        ],
    )
    if result_root != inspected.slot_path.parent:
        raise ShadowPredictionError(
            "Le repertoire derive du slot fantome est incoherent."
        )

    try:
        inspected.slot_path.mkdir(exist_ok=False)
    except FileExistsError as error:
        raise ShadowPredictionSlotConsumedError(
            "Le slot fantome a ete reserve par une autre execution."
        ) from error
    except OSError as error:
        raise ShadowPredictionError(
            "Impossible de creer le slot fantome de facon exclusive."
        ) from error

    # A partir de mkdir, le slot est consomme meme si une synchronisation ou
    # la publication de RESERVED echoue. Il ne doit donc jamais etre nettoye.
    confirmed_result_root = _require_shadow_result_root(
        project_directory=inspected.slot_path.parents[
            len(SHADOW_RESULT_ROOT_RELATIVE_PATH.parts)
        ],
    )
    if confirmed_result_root != result_root:
        raise ShadowPredictionError(
            "La racine du slot a change pendant sa reservation."
        )
    slot_mode = _lstat_mode(inspected.slot_path)
    if not isinstance(slot_mode, int) or not stat.S_ISDIR(slot_mode):
        raise ShadowPredictionError(
            "Le slot reserve n'est pas un repertoire local non symbolique."
        )
    _fsync_parent_directory(inspected.slot_path.parent)
    marker_sha256 = _publish_exclusive_verified(
        inspected.slot_path / "RESERVED",
        reserved_bytes,
    )

    return ShadowPredictionSlotReservation(
        slot_path=inspected.slot_path,
        slot_key=inspected.slot_key,
        batch_id=inspected.batch_id,
        reserved_marker=reserved_marker,
        reserved_marker_sha256=marker_sha256,
    )


def fail_shadow_prediction_slot(
    reservation: ShadowPredictionSlotReservation,
    *,
    failed_at_utc: str,
    stage: str,
    error_type: str,
    error_message: str,
    project_directory: Path = PROJECT_DIRECTORY,
) -> ShadowPredictionSlotFailure:
    """Ferme definitivement un slot deja reserve avec FAILED.json.

    La primitive n'ecrit que dans le repertoire exact porte par la preuve de
    reservation. Elle exige que RESERVED soit encore canonique et identique,
    puis publie FAILED.json sans remplacement. Aucun fichier du slot n'est
    supprime, y compris si la publication ou sa verification echoue.
    """
    if type(reservation) is not ShadowPredictionSlotReservation:
        raise ShadowPredictionError(
            "Une preuve de reservation fantome exacte est requise."
        )
    if not isinstance(reservation.slot_path, Path):
        raise ShadowPredictionError(
            "Le chemin du slot reserve doit etre un objet Path."
        )

    failed_at = _require_utc_timestamp(
        failed_at_utc,
        field="failed_at_utc",
    )
    failure_stage = _require_nonempty_text(stage, field="stage")
    failure_type = _require_nonempty_text(
        error_type,
        field="error_type",
    )
    failure_message = _require_nonempty_text(
        error_message,
        field="error_message",
    )
    slot_key = _require_sha256(reservation.slot_key, field="slot_key")
    batch_id = _require_sha256(reservation.batch_id, field="batch_id")
    reserved_sha256 = _require_sha256(
        reservation.reserved_marker_sha256,
        field="reserved_marker_sha256",
    )

    reserved = reservation.reserved_marker
    if not _has_exact_keys(reserved, _RESERVED_MARKER_KEYS):
        raise ShadowPredictionError(
            "La preuve de reservation ne contient pas un marqueur RESERVED "
            "exact."
        )
    assert isinstance(reserved, dict)
    target_date = _require_date_string(
        reserved.get("target_official_date"),
        field="RESERVED.target_official_date",
    )
    target = _parse_target_official_date(target_date)
    if (
        target.year != EXPECTED_TARGET_SEASON
        or target < EXPECTED_SHADOW_PROTOCOL_REGISTERED_ON
    ):
        raise ShadowPredictionError(
            "La preuve RESERVED ne vise pas une date admissible du "
            "protocole fantome v2."
        )
    reserved_at = _require_utc_timestamp(
        reserved.get("reserved_at_utc"),
        field="RESERVED.reserved_at_utc",
    )
    protocol_hash = _require_sha256(
        reserved.get("shadow_protocol_sha256"),
        field="RESERVED.shadow_protocol_sha256",
    )
    manifest_hash = _require_sha256(
        reserved.get("execution_manifest_sha256"),
        field="RESERVED.execution_manifest_sha256",
    )
    runtime_commit = _require_git_commit(
        reserved.get("runtime_code_commit"),
        field="RESERVED.runtime_code_commit",
    )
    if protocol_hash != EXPECTED_SHADOW_PROTOCOL_SHA256:
        raise ShadowPredictionError(
            "FAILED.json exige l'empreinte du protocole fantome v2 fige."
        )
    expected_slot_key = build_slot_key(
        shadow_protocol_sha256=protocol_hash,
        target_official_date=target_date,
    )
    expected_batch_id = build_batch_id(
        slot_key=expected_slot_key,
        execution_manifest_sha256=manifest_hash,
        model_artifact_sha256=EXPECTED_MODEL_ARTIFACT_SHA256,
    )
    expected_reserved: dict[str, Any] = {
        "marker_schema_version": 1,
        "batch_id": expected_batch_id,
        "slot_key": expected_slot_key,
        "target_official_date": target_date,
        "reserved_at_utc": reserved_at,
        "shadow_protocol_sha256": protocol_hash,
        "execution_manifest_sha256": manifest_hash,
        "runtime_code_commit": runtime_commit,
    }
    if (
        not _valid_reserved_marker(
            reserved,
            batch_id=expected_batch_id,
            slot_key=expected_slot_key,
            target_official_date=target_date,
            shadow_protocol_sha256=protocol_hash,
            execution_manifest_sha256=manifest_hash,
        )
        or reserved != expected_reserved
        or slot_key != expected_slot_key
        or batch_id != expected_batch_id
    ):
        raise ShadowPredictionError(
            "La preuve de reservation ne correspond pas aux identifiants "
            "figes du slot."
        )
    reserved_bytes = _canonical_json_file_bytes(expected_reserved)
    if hashlib.sha256(reserved_bytes).hexdigest() != reserved_sha256:
        raise ShadowPredictionError(
            "L'empreinte de la preuve RESERVED est incoherente."
        )
    if failed_at < reserved_at:
        raise ShadowPredictionError(
            "failed_at_utc ne peut pas preceder RESERVED.reserved_at_utc."
        )

    failed_marker: dict[str, Any] = {
        "marker_schema_version": 1,
        "batch_id": batch_id,
        "slot_key": slot_key,
        "target_official_date": target_date,
        "failed_at_utc": failed_at,
        "stage": failure_stage,
        "error_type": failure_type,
        "error_message": failure_message,
        "shadow_protocol_sha256": protocol_hash,
        "execution_manifest_sha256": manifest_hash,
        "runtime_code_commit": runtime_commit,
    }
    if not _has_exact_keys(failed_marker, _FAILED_MARKER_KEYS):
        raise AssertionError("Schema interne FAILED.json incoherent.")
    failed_bytes = _canonical_json_file_bytes(failed_marker)

    result_root = _require_shadow_result_root(
        project_directory=project_directory,
    )
    expected_slot_path = result_root / target_date
    if reservation.slot_path != expected_slot_path:
        raise ShadowPredictionError(
            "La preuve de reservation ne vise pas le slot attendu."
        )
    slot_mode = _lstat_mode(expected_slot_path)
    if not isinstance(slot_mode, int) or not stat.S_ISDIR(slot_mode):
        raise ShadowPredictionError(
            "Le slot reserve doit rester un repertoire local non "
            "symbolique."
        )

    for terminal_name in ("COMPLETED", "FAILED.json"):
        if _lstat_mode(expected_slot_path / terminal_name) is not _PATH_MISSING:
            raise ShadowPredictionSlotConsumedError(
                "Le slot fantome possede deja un marqueur terminal et ne "
                "peut pas etre modifie."
            )

    persisted_reserved = _read_canonical_json_object(
        expected_slot_path / "RESERVED"
    )
    if persisted_reserved is None:
        raise ShadowPredictionError(
            "Le slot ne contient pas la preuve RESERVED canonique attendue."
        )
    persisted_marker, persisted_bytes = persisted_reserved
    if (
        persisted_marker != expected_reserved
        or persisted_bytes != reserved_bytes
        or hashlib.sha256(persisted_bytes).hexdigest() != reserved_sha256
    ):
        raise ShadowPredictionError(
            "La preuve RESERVED persistee ne correspond pas a la "
            "reservation fournie."
        )

    try:
        failed_sha256 = _publish_exclusive_verified(
            expected_slot_path / "FAILED.json",
            failed_bytes,
        )
    except ShadowPublicationConflictError as error:
        raise ShadowPredictionSlotConsumedError(
            "Le slot fantome a deja ete ferme par une autre execution."
        ) from error

    return ShadowPredictionSlotFailure(
        slot_path=expected_slot_path,
        failed_marker=failed_marker,
        failed_marker_sha256=failed_sha256,
    )


def _read_frozen_protocol(
    project_directory: Path,
) -> tuple[dict[str, Any], str]:
    """Lit uniquement le chemin local impose, controle avant decodage."""
    project = Path(project_directory).expanduser().resolve()
    if not project.is_dir():
        raise ShadowPredictionError(
            f"Dossier du projet introuvable : {project}."
        )

    protocol_path = project.joinpath(*SHADOW_PROTOCOL_RELATIVE_PATH.parts)
    if protocol_path.is_symlink():
        raise ShadowPredictionError(
            "Le protocole v2 local ne peut pas etre un lien symbolique : "
            f"{SHADOW_PROTOCOL_RELATIVE_PATH.as_posix()}."
        )
    try:
        protocol_bytes = protocol_path.read_bytes()
    except OSError as error:
        raise ShadowPredictionError(
            "Protocole v2 introuvable au chemin fige "
            f"{SHADOW_PROTOCOL_RELATIVE_PATH.as_posix()}."
        ) from error

    actual_sha256 = hashlib.sha256(protocol_bytes).hexdigest()
    if actual_sha256 != EXPECTED_SHADOW_PROTOCOL_SHA256:
        raise ShadowPredictionError(
            "SHA-256 invalide pour le protocole v2 local : "
            f"{actual_sha256}, attendu {EXPECTED_SHADOW_PROTOCOL_SHA256}."
        )

    try:
        decoded = protocol_bytes.decode("utf-8")
        payload = json.loads(
            decoded,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ShadowPredictionError(
            "Le protocole v2 fige n'est pas un objet JSON UTF-8 valide."
        ) from error
    protocol = dict(
        _require_mapping(payload, description="le protocole v2")
    )
    return protocol, actual_sha256


def _validate_protocol_contract(protocol: Mapping[str, Any]) -> None:
    """Valide les verrous necessaires au seul mode apercu."""
    _require_exact(
        protocol,
        "shadow_prediction_protocol_version",
        EXPECTED_SHADOW_PROTOCOL_VERSION,
        context="protocol",
    )
    _require_exact(
        protocol,
        "registered_on",
        EXPECTED_SHADOW_PROTOCOL_REGISTERED_ON.isoformat(),
        context="protocol",
    )
    _require_exact(
        protocol,
        "status",
        EXPECTED_SHADOW_PROTOCOL_STATUS,
        context="protocol",
    )
    _require_exact(
        protocol,
        "first_shadow_prediction_created_at_registration",
        False,
        context="protocol",
    )

    freeze = _require_mapping(
        protocol.get("execution_freeze"),
        description="protocol.execution_freeze",
    )
    _require_exact(
        freeze,
        "shadow_service_module_path_exact",
        EXPECTED_SERVICE_MODULE_PATH,
        context="protocol.execution_freeze",
    )
    _require_exact(
        freeze,
        "shadow_protocol_path_exact",
        SHADOW_PROTOCOL_RELATIVE_PATH.as_posix(),
        context="protocol.execution_freeze",
    )
    _require_exact(
        freeze,
        "default_mode",
        EXPECTED_PREVIEW_MODE,
        context="protocol.execution_freeze",
    )
    for key in (
        "preview_must_not_deserialize_model",
        "preview_must_not_call_predict_proba",
        "preview_must_not_reserve_output_slot",
        "preview_must_not_create_output_files",
    ):
        _require_exact(
            freeze,
            key,
            True,
            context="protocol.execution_freeze",
        )

    operating = _require_mapping(
        protocol.get("operating_mode"),
        description="protocol.operating_mode",
    )
    for key, expected in (
        ("strictly_prospective", True),
        ("historical_backfill_allowed", False),
        ("target_season_exact", EXPECTED_TARGET_SEASON),
        ("future_other_seasons_require_new_registered_protocol", True),
        ("time_basis", "UTC"),
    ):
        _require_exact(
            operating,
            key,
            expected,
            context="protocol.operating_mode",
        )

    candidates = _require_mapping(
        protocol.get("candidate_games"),
        description="protocol.candidate_games",
    )
    _require_exact(
        candidates,
        "season_must_equal",
        EXPECTED_TARGET_SEASON,
        context="protocol.candidate_games",
    )

    lineage = _require_mapping(
        protocol.get("validated_lineage"),
        description="protocol.validated_lineage",
    )
    _validate_declared_file(
        lineage,
        "model_artifact",
        expected_path=EXPECTED_MODEL_ARTIFACT_PATH,
        expected_sha256=EXPECTED_MODEL_ARTIFACT_SHA256,
    )
    _validate_declared_file(
        lineage,
        "artifact_manifest",
        expected_path=EXPECTED_ARTIFACT_MANIFEST_PATH,
        expected_sha256=EXPECTED_ARTIFACT_MANIFEST_SHA256,
    )
    _validate_declared_file(
        lineage,
        "model_protocol",
        expected_path=EXPECTED_MODEL_PROTOCOL_PATH,
        expected_sha256=EXPECTED_MODEL_PROTOCOL_SHA256,
    )
    historical = _require_mapping(
        lineage.get("historical_dataset_lineage_only"),
        description=(
            "protocol.validated_lineage.historical_dataset_lineage_only"
        ),
    )
    _require_exact(
        historical,
        "must_not_be_loaded_for_shadow_prediction",
        True,
        context=(
            "protocol.validated_lineage.historical_dataset_lineage_only"
        ),
    )


def _validate_declared_file(
    lineage: Mapping[str, Any],
    key: str,
    *,
    expected_path: str,
    expected_sha256: str,
) -> None:
    declaration = _require_mapping(
        lineage.get(key),
        description=f"protocol.validated_lineage.{key}",
    )
    context = f"protocol.validated_lineage.{key}"
    _require_exact(
        declaration,
        "path",
        expected_path,
        context=context,
    )
    _require_exact(
        declaration,
        "sha256",
        expected_sha256,
        context=context,
    )


def _parse_target_official_date(value: date | str) -> date:
    """Accepte une date ou une chaine ISO canonique, jamais un instant."""
    if isinstance(value, datetime):
        raise ShadowPredictionError(
            "target_official_date doit etre une date sans heure."
        )
    if isinstance(value, date):
        return value
    if not isinstance(value, str) or not _CANONICAL_DATE_PATTERN.fullmatch(
        value
    ):
        raise ShadowPredictionError(
            "target_official_date doit respecter exactement YYYY-MM-DD."
        )
    try:
        parsed = date.fromisoformat(value)
    except ValueError as error:
        raise ShadowPredictionError(
            f"target_official_date invalide : {value!r}."
        ) from error
    if parsed.isoformat() != value:
        raise ShadowPredictionError(
            "target_official_date doit etre une date ISO canonique."
        )
    return parsed


def preview_shadow_prediction(
    target_official_date: date | str,
    *,
    project_directory: Path = PROJECT_DIRECTORY,
) -> ShadowPredictionPreview:
    """Valide le protocole et la date sans ouvrir aucune entree officielle."""
    protocol, protocol_sha256 = _read_frozen_protocol(project_directory)
    _validate_protocol_contract(protocol)
    target = _parse_target_official_date(target_official_date)

    if target.year != EXPECTED_TARGET_SEASON:
        raise ShadowPredictionError(
            "Saison cible invalide : "
            f"{target.year}, attendu {EXPECTED_TARGET_SEASON}."
        )
    if target < EXPECTED_SHADOW_PROTOCOL_REGISTERED_ON:
        raise ShadowPredictionError(
            "La date cible ne peut pas preceder l'enregistrement du "
            f"protocole v2 ({EXPECTED_SHADOW_PROTOCOL_REGISTERED_ON})."
        )

    return ShadowPredictionPreview(
        mode=EXPECTED_PREVIEW_MODE,
        shadow_protocol_version=EXPECTED_SHADOW_PROTOCOL_VERSION,
        shadow_protocol_path=SHADOW_PROTOCOL_RELATIVE_PATH.as_posix(),
        shadow_protocol_sha256=protocol_sha256,
        shadow_protocol_registered_on=(
            EXPECTED_SHADOW_PROTOCOL_REGISTERED_ON.isoformat()
        ),
        shadow_protocol_status=EXPECTED_SHADOW_PROTOCOL_STATUS,
        target_official_date=target.isoformat(),
        target_season=target.year,
        target_validation_scope="STATIC_PROTOCOL_ONLY",
        execution_ready=False,
        execution_manifest_read=False,
        activation_read=False,
        model_artifact_read=False,
        model_deserialized=False,
        sqlite_read=False,
        network_request_performed=False,
        output_slot_reserved=False,
        output_files_created=False,
        predictions_computed=False,
    )


def _build_argument_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
        description=(
            "Valide localement le protocole fantome MLB v2 et une date "
            "cible. Ce jalon est un apercu sans modele ni prediction."
        )
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Affiche un JSON deterministe; aucun mode d'execution n'existe ici."""
    parser = _build_argument_parser()
    parser.add_argument(
        "--target-official-date",
        required=True,
        metavar="YYYY-MM-DD",
        help="Date officielle MLB cible, obligatoirement en saison 2026.",
    )
    arguments = parser.parse_args(argv)
    try:
        preview = preview_shadow_prediction(arguments.target_official_date)
    except ShadowPredictionError as error:
        parser.error(str(error))
    print(preview.to_canonical_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
