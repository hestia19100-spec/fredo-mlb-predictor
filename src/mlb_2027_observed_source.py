"""Relie les candidats MLB 2027 aux observations officielles et au délai T−60.

Ce module reste interne : il n'émet ni publication ni certification distante.
L'appelant doit archiver les octets et revérifier lui-même la réponse distante
qui atteste la publication effective des probabilités.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
import hashlib
import json
from urllib.parse import parse_qsl, urlsplit

from src.mlb_api import (
    MLBAPIError,
    MLB_SCHEDULE_URL,
    ScheduleFetchResult,
    _parse_http_date,
    _parse_schedule_payload,
)
from src.mlb_2027_reference_predictor import (
    ReferenceCalculation,
    calculate_reference_probabilities,
)
from src.mlb_2027_reference_readiness import (
    PriorFinalGame,
    UpcomingGame,
)
from src.mlb_2027_timing import is_prospectively_on_time


class MLB2027SourceError(ValueError):
    """La provenance ou la chronologie ne permet pas de calcul sûr."""


@dataclass(frozen=True, slots=True)
class ObservedSourceBundle:
    target_date: date
    observations: tuple[ScheduleFetchResult, ...]
    upcoming_games: tuple[UpcomingGame, ...]
    prior_final_games: tuple[PriorFinalGame, ...]
    receipt_bytes: bytes
    receipt_sha256: str
    latest_received_at_utc: datetime


def _utc(value: object, label: str) -> datetime:
    if type(value) is not str or not value.endswith("Z"):
        raise MLB2027SourceError(f"{label} doit être un instant UTC canonique.")
    try:
        instant = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise MLB2027SourceError(f"{label} est invalide.") from error
    if (
        instant.utcoffset() != timedelta(0)
        or instant.microsecond
        or instant.strftime("%Y-%m-%dT%H:%M:%SZ") != value
    ):
        raise MLB2027SourceError(f"{label} doit être un instant UTC à la seconde.")
    return instant


def _no_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise MLB2027SourceError("Clé JSON dupliquée dans la réponse MLB.")
        result[key] = value
    return result


def _validate_observation(
    observation: ScheduleFetchResult,
) -> tuple[tuple[object, ...], datetime, dict[str, object]]:
    if type(observation) is not ScheduleFetchResult:
        raise MLB2027SourceError("Observation MLB non reconnue.")
    if (
        type(observation.start_date) is not date
        or type(observation.end_date) is not date
        or observation.start_date > observation.end_date
        or (observation.end_date - observation.start_date).days >= 31
        or observation.game_types != ("R",)
    ):
        raise MLB2027SourceError("Période MLB invalide ou non régulière.")
    expected_parameters = {
        "sportId": 1,
        "startDate": observation.start_date.isoformat(),
        "endDate": observation.end_date.isoformat(),
        "gameTypes": "R",
        "hydrate": "probablePitcher",
    }
    if observation.request_parameters != expected_parameters:
        raise MLB2027SourceError("Paramètres de collecte MLB inattendus.")
    url = observation.response_effective_url
    if type(url) is not str:
        raise MLB2027SourceError("URL effective MLB absente.")
    parsed = urlsplit(url)
    try:
        pairs = parse_qsl(parsed.query, keep_blank_values=True, strict_parsing=True)
    except ValueError as error:
        raise MLB2027SourceError("URL effective MLB invalide.") from error
    if (
        f"{parsed.scheme}://{parsed.netloc}{parsed.path}" != MLB_SCHEDULE_URL
        or parsed.fragment
        or len(pairs) != len(expected_parameters)
        or dict(pairs) != {key: str(value) for key, value in expected_parameters.items()}
    ):
        raise MLB2027SourceError("La réponse ne provient pas de l'URL MLB attendue.")
    if (
        type(observation.response_status_code) is not int
        or observation.response_status_code != 200
        or type(observation.response_redirect_count) is not int
        or observation.response_redirect_count != 0
        or type(observation.raw_content) is not bytes
        or not observation.raw_content
    ):
        raise MLB2027SourceError("Réponse MLB absente, redirigée ou non réussie.")
    body_sha = hashlib.sha256(observation.raw_content).hexdigest()
    if observation.response_body_sha256 != body_sha:
        raise MLB2027SourceError("Empreinte du corps MLB incohérente.")
    try:
        http_date = _parse_http_date(observation.mlb_http_date_header_raw)
    except Exception as error:
        raise MLB2027SourceError("Date HTTP MLB invalide.") from error
    if http_date != observation.mlb_http_date_utc:
        raise MLB2027SourceError("Date HTTP MLB incohérente.")
    received = _utc(
        observation.mlb_http_response_received_at_utc, "réception MLB"
    )
    if abs((received - _utc(http_date, "Date HTTP MLB")).total_seconds()) > 300:
        raise MLB2027SourceError("Horloge locale trop éloignée de MLB.")
    try:
        payload = json.loads(
            observation.raw_content.decode("utf-8"),
            object_pairs_hook=_no_duplicate_pairs,
        )
        if type(payload) is not dict:
            raise MLB2027SourceError("Réponse MLB non structurée.")
        games = tuple(_parse_schedule_payload(
            payload, reject_duplicate_game_ids=True,
        ))
    except (MLBAPIError, UnicodeError, ValueError, TypeError, KeyError) as error:
        raise MLB2027SourceError("Réponse MLB illisible ou ambiguë.") from error
    if games != observation.games:
        raise MLB2027SourceError("Les matchs ne correspondent pas aux octets MLB.")
    for game in games:
        try:
            official = date.fromisoformat(game.official_date)
        except ValueError as error:
            raise MLB2027SourceError("Date officielle MLB invalide.") from error
        if not observation.start_date <= official <= observation.end_date:
            raise MLB2027SourceError("Match hors de la période MLB demandée.")
    proof = {
        "start_date": observation.start_date.isoformat(),
        "end_date": observation.end_date.isoformat(),
        "effective_url": url,
        "response_body_sha256": body_sha,
        "mlb_http_date_utc": http_date,
        "response_received_at_utc": received.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    return games, received, proof


def _game_start(value: object) -> datetime:
    return _utc(value, "début du match MLB")


def verify_2027_observed_source(
    target_date: date,
    observations: tuple[ScheduleFetchResult, ...],
) -> ObservedSourceBundle:
    """Vérifie la couverture intégrale 2027 et les réponses HTTP avant calcul.

    La collecte observée doit couvrir sans trou du 1er janvier au jour cible.
    Cela évite de déclarer dix finales à partir d'un historique tronqué.
    """
    if type(target_date) is not date or target_date.year != 2027:
        raise MLB2027SourceError("Seule la saison officielle 2027 est admise.")
    if type(observations) is not tuple or not observations:
        raise MLB2027SourceError("Observations MLB manquantes.")
    ordered = sorted(observations, key=lambda row: row.start_date)
    expected_start = date(2027, 1, 1)
    games_by_id: dict[int, object] = {}
    receipts: list[dict[str, object]] = []
    latest: datetime | None = None
    for observation in ordered:
        if (
            observation.start_date != expected_start
            or observation.end_date > target_date
        ):
            raise MLB2027SourceError("Couverture MLB incomplète ou chevauchante.")
        games, received, proof = _validate_observation(observation)
        expected_start = observation.end_date + timedelta(days=1)
        latest = received if latest is None else max(latest, received)
        receipts.append(proof)
        for game in games:
            if game.game_id in games_by_id:
                raise MLB2027SourceError("Identifiant de match MLB dupliqué.")
            games_by_id[game.game_id] = game
    if expected_start != target_date + timedelta(days=1) or latest is None:
        raise MLB2027SourceError("Le jour cible n'est pas couvert.")

    upcoming: list[UpcomingGame] = []
    prior: list[PriorFinalGame] = []
    for game in games_by_id.values():
        if game.season != 2027 or game.game_type != "R":
            raise MLB2027SourceError("Saison ou type de match MLB inattendu.")
        official = date.fromisoformat(game.official_date)
        if official == target_date:
            if (
                game.status_code not in {"S", "P"}
                or game.status_detail.upper() not in {"SCHEDULED", "PRE-GAME"}
                or game.away_score is not None
                or game.home_score is not None
            ):
                raise MLB2027SourceError("Un match du jour a déjà commencé ou changé d'état.")
            upcoming.append(UpcomingGame(
                game_id=game.game_id, official_date=official, game_type="R",
                away_team_id=game.away_team_id,
                home_team_id=game.home_team_id,
                scheduled_start_utc=_game_start(game.game_datetime_utc),
                status="SCHEDULED" if game.status_detail.upper() == "SCHEDULED"
                else "PRE_GAME",
            ))
        elif (
            game.status_code == "F"
            and game.status_detail.upper() in {"FINAL", "GAME OVER", "COMPLETED EARLY"}
        ):
            if (
                type(game.away_score) is not int
                or type(game.home_score) is not int
                or min(game.away_score, game.home_score) < 0
                or game.away_score == game.home_score
            ):
                raise MLB2027SourceError("Score final MLB invalide.")
            prior.append(PriorFinalGame(
                game_id=game.game_id, official_date=official, game_type="R",
                away_team_id=game.away_team_id,
                home_team_id=game.home_team_id,
                away_score=game.away_score, home_score=game.home_score,
                status="FINAL",
            ))
    if not upcoming:
        raise MLB2027SourceError("Aucun match MLB régulier avant match pour cette date.")
    receipt_bytes = (
        json.dumps(
            {"schema_version": 1, "target_date": target_date.isoformat(),
             "observations": receipts},
            sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        ).encode("utf-8") + b"\n"
    )
    return ObservedSourceBundle(
        target_date=target_date,
        observations=tuple(ordered),
        upcoming_games=tuple(upcoming),
        prior_final_games=tuple(prior),
        receipt_bytes=receipt_bytes,
        receipt_sha256=hashlib.sha256(receipt_bytes).hexdigest(),
        latest_received_at_utc=latest,
    )


def candidate_has_prospective_timing(
    source: ObservedSourceBundle,
    calculation: ReferenceCalculation,
    *,
    issued_at_utc: datetime,
    completed_at_utc: datetime,
    remote_http_date_utc: datetime,
) -> bool:
    """Contrôle les horloges; ne certifie pas la publication distante.

    La preuve distante doit être obtenue et vérifiée par l'orchestrateur.
    Un candidat tardif est refusé, jamais antidaté.
    """
    if (
        type(source) is not ObservedSourceBundle
        or type(calculation) is not ReferenceCalculation
        or calculation.readiness.target_date != source.target_date
        or calculation.readiness.source_receipt_sha256 != source.receipt_sha256
        or {game.game_id for game in calculation.readiness.games}
        != {game.game_id for game in source.upcoming_games}
    ):
        raise MLB2027SourceError("Calcul et source MLB non liés.")
    expected = calculate_reference_probabilities(
        source.target_date, source.upcoming_games, source.prior_final_games,
        source_receipt_sha256=source.receipt_sha256,
    )
    if calculation != expected:
        raise MLB2027SourceError("Probabilités non liées au modèle figé.")
    if source != verify_2027_observed_source(
        source.target_date, source.observations,
    ):
        raise MLB2027SourceError("Résumé de source MLB altéré.")
    if (
        not isinstance(issued_at_utc, datetime)
        or issued_at_utc.tzinfo is None
        or issued_at_utc.utcoffset() != timedelta(0)
        or source.latest_received_at_utc > issued_at_utc
    ):
        raise MLB2027SourceError("La source MLB n'était pas disponible à l'émission.")
    return is_prospectively_on_time(
        calculation.readiness.lead_window,
        issued_at_utc=issued_at_utc,
        completed_at_utc=completed_at_utc,
        remote_http_date_utc=remote_http_date_utc,
    )
