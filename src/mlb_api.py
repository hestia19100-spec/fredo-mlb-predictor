"""Client contrôlé pour récupérer le calendrier depuis l’API MLB."""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from datetime import date, datetime, timezone
from email.utils import format_datetime, parsedate_to_datetime
import hashlib
import json
from typing import Any
from urllib.parse import parse_qsl, urlsplit

import requests


MLB_SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"
MLB_SPORT_ID = 1
REQUEST_TIMEOUT_SECONDS = 30
MAX_SCHEDULE_RANGE_DAYS = 31
DISPLAY_LIMIT = 20

FINAL_STATUS_CODES = frozenset({"F"})
FINAL_STATUS_DETAILS = frozenset(
    {
        "FINAL",
        "GAME OVER",
        "COMPLETED EARLY",
    }
)
POSTPONED_STATUS_DETAILS = frozenset({"POSTPONED"})


class MLBAPIError(RuntimeError):
    """Erreur compréhensible liée à l’API MLB."""


class MLBAPIRetryableError(MLBAPIError):
    """Erreur temporaire autorisant une nouvelle tentative."""


@dataclass(frozen=True, slots=True)
class ScheduledGame:
    """Représentation contrôlée d’un match reçu depuis MLB."""

    game_id: int
    season: int
    official_date: str
    game_datetime_utc: str | None
    game_type: str
    status_code: str
    status_detail: str
    away_team_id: int
    away_team_name: str
    home_team_id: int
    home_team_name: str
    away_score: int | None
    home_score: int | None
    venue_id: int | None
    venue_name: str | None
    doubleheader: str | None
    game_number: int | None
    away_probable_pitcher_id: int | None
    away_probable_pitcher_name: str | None
    home_probable_pitcher_id: int | None
    home_probable_pitcher_name: str | None
    abstract_state: str = ""


@dataclass(frozen=True, slots=True)
class ScheduleFetchResult:
    """Résultat auditable d’une récupération par période."""

    start_date: date
    end_date: date
    game_types: tuple[str, ...]
    request_parameters: dict[str, object]
    raw_content: bytes
    games: tuple[ScheduledGame, ...]
    response_effective_url: str | None = None
    response_status_code: int | None = None
    response_redirect_count: int | None = None
    mlb_http_date_header_raw: str | None = None
    mlb_http_date_utc: str | None = None
    mlb_http_response_received_at_utc: str | None = None
    response_body_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class _ScheduleHTTPResponse:
    """Réponse HTTP décodée avec ses preuves temporelles."""

    raw_content: bytes
    payload: dict[str, Any]
    response_effective_url: str
    response_status_code: int
    response_redirect_count: int
    mlb_http_date_header_raw: str | None
    mlb_http_date_utc: str | None
    response_received_at_utc: str
    response_body_sha256: str


def _utc_now() -> datetime:
    """Retourne l'heure UTC utilisée comme preuve locale."""
    return datetime.now(timezone.utc)


def _format_utc_seconds(value: datetime) -> str:
    """Normalise un instant UTC au format RFC3339 à la seconde."""
    if value.tzinfo is None or value.utcoffset() is None:
        raise MLBAPIError("Un horodatage UTC avec fuseau est obligatoire.")

    return (
        value.astimezone(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _parse_http_date(value: object) -> str:
    """Valide l'en-tête HTTP Date renvoyé par MLB."""
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
    ):
        raise MLBAPIError("L'en-tête HTTP Date de MLB est absent.")

    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise MLBAPIError(
            "L'en-tête HTTP Date de MLB est invalide."
        ) from error

    if (
        parsed.tzinfo is None
        or parsed.utcoffset() is None
        or parsed.utcoffset().total_seconds() != 0
    ):
        raise MLBAPIError("L'en-tête HTTP Date de MLB n'est pas UTC.")

    canonical_value = format_datetime(
        parsed.astimezone(timezone.utc),
        usegmt=True,
    )
    if value != canonical_value:
        raise MLBAPIError(
            "L'en-tête HTTP Date de MLB n'est pas au format "
            "IMF-fixdate GMT canonique."
        )

    return _format_utc_seconds(parsed)


def _validate_observed_effective_url(
    effective_url: object,
    parameters: dict[str, object],
) -> str:
    """Valide l'URL réellement servie pour une observation shadow."""
    if not isinstance(effective_url, str) or not effective_url:
        raise MLBAPIError("L'URL effective de la réponse MLB est absente.")

    parsed_url = urlsplit(effective_url)
    if (
        parsed_url.scheme != "https"
        or parsed_url.netloc != "statsapi.mlb.com"
        or parsed_url.path != "/api/v1/schedule"
        or parsed_url.fragment
    ):
        raise MLBAPIError("L'URL effective de la réponse MLB est invalide.")

    try:
        query_pairs = parse_qsl(
            parsed_url.query,
            keep_blank_values=True,
            strict_parsing=True,
        )
    except ValueError as error:
        raise MLBAPIError(
            "La requête effective MLB contient une query invalide."
        ) from error

    query_names = [name for name, _ in query_pairs]
    if len(query_names) != len(set(query_names)):
        raise MLBAPIError(
            "La requête effective MLB contient un paramètre en doublon."
        )

    actual_query = dict(query_pairs)
    expected_query = {
        str(name): str(value)
        for name, value in parameters.items()
    }
    if actual_query != expected_query:
        raise MLBAPIError(
            "Les paramètres de l'URL effective MLB sont inattendus."
        )

    return effective_url


def _validate_observed_clock(
    *,
    mlb_http_date_utc: str,
    response_received_at: datetime,
) -> None:
    """Refuse une horloge locale trop éloignée de la preuve MLB."""
    parsed_http_date = datetime.fromisoformat(
        mlb_http_date_utc.replace("Z", "+00:00")
    )
    normalized_received_at = (
        response_received_at
        .astimezone(timezone.utc)
        .replace(microsecond=0)
    )
    clock_skew_seconds = (
        normalized_received_at
        - parsed_http_date
    ).total_seconds()
    if abs(clock_skew_seconds) > 300:
        raise MLBAPIError(
            "L'horloge locale diffère de plus de 300 secondes "
            "de l'en-tête HTTP Date de MLB."
        )


def _optional_int(value: Any) -> int | None:
    """Convertit une valeur facultative en nombre entier."""
    if value is None:
        return None

    return int(value)


def _parse_probable_pitcher(
    team_data: dict[str, Any],
) -> tuple[int | None, str | None]:
    """Extrait le lanceur probable lorsqu’il est annoncé."""
    probable_pitcher = team_data.get("probablePitcher")

    if not isinstance(probable_pitcher, dict):
        return None, None

    pitcher_id = _optional_int(
        probable_pitcher.get("id")
    )

    pitcher_name = (
        str(probable_pitcher["fullName"])
        if probable_pitcher.get("fullName") is not None
        else None
    )

    return pitcher_id, pitcher_name


def _parse_game(
    raw_game: dict[str, Any],
) -> ScheduledGame:
    """Transforme un match MLB brut en structure contrôlée."""
    try:
        away_data = raw_game["teams"]["away"]
        home_data = raw_game["teams"]["home"]
        away_team = away_data["team"]
        home_team = home_data["team"]
        status = raw_game["status"]
        venue = raw_game.get("venue") or {}

        (
            away_probable_pitcher_id,
            away_probable_pitcher_name,
        ) = _parse_probable_pitcher(
            away_data
        )

        (
            home_probable_pitcher_id,
            home_probable_pitcher_name,
        ) = _parse_probable_pitcher(
            home_data
        )

        return ScheduledGame(
            game_id=int(raw_game["gamePk"]),
            season=int(raw_game["season"]),
            official_date=str(
                raw_game["officialDate"]
            ),
            game_datetime_utc=(
                str(raw_game["gameDate"])
                if raw_game.get("gameDate") is not None
                else None
            ),
            game_type=str(
                raw_game["gameType"]
            ),
            status_code=str(
                status.get("statusCode")
                or status.get("codedGameState")
                or "UNKNOWN"
            ),
            status_detail=str(
                status["detailedState"]
            ),
            away_team_id=int(
                away_team["id"]
            ),
            away_team_name=str(
                away_team["name"]
            ),
            home_team_id=int(
                home_team["id"]
            ),
            home_team_name=str(
                home_team["name"]
            ),
            away_score=_optional_int(
                away_data.get("score")
            ),
            home_score=_optional_int(
                home_data.get("score")
            ),
            venue_id=_optional_int(
                venue.get("id")
            ),
            venue_name=(
                str(venue["name"])
                if venue.get("name") is not None
                else None
            ),
            doubleheader=(
                str(raw_game["doubleHeader"])
                if raw_game.get("doubleHeader") is not None
                else None
            ),
            game_number=_optional_int(
                raw_game.get("gameNumber")
            ),
            away_probable_pitcher_id=(
                away_probable_pitcher_id
            ),
            away_probable_pitcher_name=(
                away_probable_pitcher_name
            ),
            home_probable_pitcher_id=(
                home_probable_pitcher_id
            ),
            home_probable_pitcher_name=(
                home_probable_pitcher_name
            ),
            abstract_state=str(
                status.get("abstractGameState")
                or ""
            ),
        )

    except (
        KeyError,
        TypeError,
        ValueError,
    ) as error:
        game_reference = raw_game.get(
            "gamePk",
            "inconnu",
        )

        raise MLBAPIError(
            "Réponse MLB incomplète pour "
            f"le match {game_reference}."
        ) from error


def _game_identity(
    game: ScheduledGame,
) -> tuple[int, str, int, int]:
    """Retourne les éléments qui ne doivent jamais changer."""
    return (
        game.season,
        game.game_type,
        game.away_team_id,
        game.home_team_id,
    )


def _is_final_game(
    game: ScheduledGame,
) -> bool:
    """Indique si une occurrence décrit un match terminé."""
    status_code = (
        game.status_code
        .strip()
        .upper()
    )

    status_detail = (
        game.status_detail
        .strip()
        .upper()
    )

    return (
        status_code in FINAL_STATUS_CODES
        or status_detail in FINAL_STATUS_DETAILS
    )


def _is_postponed_game(
    game: ScheduledGame,
) -> bool:
    """Indique si une occurrence décrit un match reporté."""
    status_detail = (
        game.status_detail
        .strip()
        .upper()
    )

    return (
        status_detail
        in POSTPONED_STATUS_DETAILS
    )


def _differ_only_by_game_datetime(
    existing_game: ScheduledGame,
    candidate_game: ScheduledGame,
) -> bool:
    """
    Autorise uniquement une différence d’horaire.

    Cette situation existe lorsqu’un match suspendu
    est terminé le lendemain tout en conservant le même
    identifiant, la même date officielle et le même score.
    """
    normalized_existing = replace(
        existing_game,
        game_datetime_utc="",
    )

    normalized_candidate = replace(
        candidate_game,
        game_datetime_utc="",
    )

    return (
        normalized_existing
        == normalized_candidate
    )


def _differ_only_by_postponement_schedule(
    existing_game: ScheduledGame,
    candidate_game: ScheduledGame,
) -> bool:
    """
    Autorise uniquement les changements de reprogrammation.

    MLB peut publier plusieurs occurrences reportées du même
    match avec un nouvel horaire, un nouveau numéro de match
    ou un statut de double programme modifié.
    """
    normalized_existing = replace(
        existing_game,
        game_datetime_utc="",
        doubleheader=None,
        game_number=None,
    )

    normalized_candidate = replace(
        candidate_game,
        game_datetime_utc="",
        doubleheader=None,
        game_number=None,
    )

    return (
        normalized_existing
        == normalized_candidate
    )


def _select_canonical_game(
    existing_game: ScheduledGame,
    candidate_game: ScheduledGame,
) -> ScheduledGame:
    """Fusionne deux occurrences cohérentes d’un même match."""
    if (
        _game_identity(existing_game)
        != _game_identity(candidate_game)
    ):
        raise MLBAPIError(
            "Occurrences contradictoires pour "
            "le match MLB "
            f"{candidate_game.game_id}."
        )

    if existing_game == candidate_game:
        return existing_game

    existing_is_final = _is_final_game(
        existing_game
    )

    candidate_is_final = _is_final_game(
        candidate_game
    )

    if (
        existing_is_final
        and candidate_is_final
    ):
        if _differ_only_by_game_datetime(
            existing_game,
            candidate_game,
        ):
            return min(
                (
                    existing_game,
                    candidate_game,
                ),
                key=lambda game: (
                    game.game_datetime_utc is None,
                    game.game_datetime_utc or "",
                ),
            )

        raise MLBAPIError(
            "Occurrences contradictoires pour "
            "le match MLB "
            f"{candidate_game.game_id}."
        )

    if existing_is_final != candidate_is_final:
        if candidate_is_final:
            return candidate_game

        return existing_game

    existing_is_postponed = _is_postponed_game(
        existing_game
    )

    candidate_is_postponed = _is_postponed_game(
        candidate_game
    )

    if (
        existing_is_postponed
        and candidate_is_postponed
    ):
        if _differ_only_by_postponement_schedule(
            existing_game,
            candidate_game,
        ):
            return max(
                (
                    existing_game,
                    candidate_game,
                ),
                key=lambda game: (
                    game.game_datetime_utc is not None,
                    game.game_datetime_utc or "",
                ),
            )

        raise MLBAPIError(
            "Occurrences contradictoires pour "
            "le match MLB "
            f"{candidate_game.game_id}."
        )

    if (
        existing_is_postponed
        != candidate_is_postponed
    ):
        if candidate_is_postponed:
            return existing_game

        return candidate_game

    raise MLBAPIError(
        "Occurrences contradictoires pour "
        "le match MLB "
        f"{candidate_game.game_id}."
    )


def _parse_schedule_payload(
    payload: dict[str, Any],
    *,
    reject_duplicate_game_ids: bool = False,
) -> list[ScheduledGame]:
    """Transforme et contrôle tous les matchs d’une réponse."""
    date_blocks = payload.get(
        "dates",
        [],
    )

    if not isinstance(
        date_blocks,
        list,
    ):
        raise MLBAPIError(
            "Liste des dates absente "
            "de la réponse MLB."
        )

    game_occurrences: list[
        ScheduledGame
    ] = []

    for date_block in date_blocks:
        if not isinstance(
            date_block,
            dict,
        ):
            raise MLBAPIError(
                "Bloc de date MLB invalide."
            )

        raw_games = date_block.get(
            "games",
            [],
        )

        if not isinstance(
            raw_games,
            list,
        ):
            raise MLBAPIError(
                "Liste des matchs MLB invalide."
            )

        for raw_game in raw_games:
            if not isinstance(
                raw_game,
                dict,
            ):
                raise MLBAPIError(
                    "Match MLB invalide."
                )

            game_occurrences.append(
                _parse_game(raw_game)
            )

    reported_total = payload.get(
        "totalGames"
    )

    if reported_total is not None:
        try:
            expected_total = int(
                reported_total
            )

        except (
            TypeError,
            ValueError,
        ) as error:
            raise MLBAPIError(
                "Le total de matchs MLB "
                "est invalide."
            ) from error

        if (
            expected_total
            != len(game_occurrences)
        ):
            raise MLBAPIError(
                "La réponse MLB semble partielle : "
                f"{len(game_occurrences)} matchs "
                f"lus sur {expected_total} annoncés."
            )

    if reject_duplicate_game_ids:
        game_ids = [game.game_id for game in game_occurrences]
        if len(game_ids) != len(set(game_ids)):
            raise MLBAPIError(
                "La réponse MLB prospective contient des identifiants "
                "de match en doublon."
            )

    canonical_games: dict[
        int,
        ScheduledGame,
    ] = {}

    for game in game_occurrences:
        existing_game = canonical_games.get(
            game.game_id
        )

        if existing_game is None:
            canonical_games[
                game.game_id
            ] = game

            continue

        canonical_games[
            game.game_id
        ] = _select_canonical_game(
            existing_game,
            game,
        )

    return list(
        canonical_games.values()
    )


def _request_schedule_response(
    parameters: dict[str, object],
    *,
    require_http_date: bool,
) -> _ScheduleHTTPResponse:
    """Télécharge et décode une réponse avec preuve temporelle."""
    headers = {
        "User-Agent": (
            "fredo-mlb-predictor/0.1"
        ),
    }

    try:
        request_options: dict[str, Any] = {
            "params": parameters,
            "headers": headers,
            "timeout": REQUEST_TIMEOUT_SECONDS,
        }
        if require_http_date:
            request_options["allow_redirects"] = False

        response = requests.get(
            MLB_SCHEDULE_URL,
            **request_options,
        )

        response_received_at = _utc_now()
        response_received_at_utc = _format_utc_seconds(
            response_received_at
        )

        response.raise_for_status()

    except (
        requests.Timeout,
        requests.ConnectionError,
    ) as error:
        raise MLBAPIRetryableError(
            "Impossible de joindre "
            f"l’API MLB : {error}"
        ) from error

    except requests.HTTPError as error:
        status_code = (
            error.response.status_code
            if error.response is not None
            else None
        )

        if status_code is None:
            error_message = (
                "L’API MLB a répondu avec "
                "une erreur HTTP."
            )
        else:
            error_message = (
                "L’API MLB a répondu avec "
                "une erreur HTTP "
                f"{status_code}."
            )

        if (
            status_code in {408, 429}
            or (
                status_code is not None
                and 500 <= status_code <= 599
            )
        ):
            raise MLBAPIRetryableError(
                error_message
            ) from error

        raise MLBAPIError(
            error_message
        ) from error

    except requests.RequestException as error:
        raise MLBAPIError(
            f"Requête MLB invalide : {error}"
        ) from error

    raw_content = response.content

    if (
        not isinstance(
            raw_content,
            bytes,
        )
        or not raw_content
    ):
        raise MLBAPIError(
            "L’API MLB a renvoyé "
            "une réponse vide."
        )

    try:
        payload = json.loads(
            raw_content
        )

    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
    ) as error:
        raise MLBAPIError(
            "L’API MLB n’a pas renvoyé "
            "un document JSON valide."
        ) from error

    if not isinstance(
        payload,
        dict,
    ):
        raise MLBAPIError(
            "Format général inattendu "
            "dans la réponse MLB."
        )

    response_body_sha256 = hashlib.sha256(raw_content).hexdigest()
    response_status_value = getattr(response, "status_code", None)
    response_history = getattr(response, "history", ())
    response_url_value = getattr(response, "url", None)

    if require_http_date:
        if type(response_status_value) is not int:
            raise MLBAPIError("Le statut HTTP de MLB est invalide.")
        if response_status_value != 200:
            raise MLBAPIError(
                "Le statut HTTP observé de MLB doit être exactement 200."
            )
        if not isinstance(response_history, (list, tuple)):
            raise MLBAPIError("L'historique de redirection MLB est invalide.")
        if len(response_history) != 0:
            raise MLBAPIError("Une redirection HTTP MLB est interdite.")

        response_effective_url = _validate_observed_effective_url(
            response_url_value,
            parameters,
        )
        response_status_code = response_status_value
        response_redirect_count = len(response_history)
    else:
        response_effective_url = (
            response_url_value
            if isinstance(response_url_value, str)
            else MLB_SCHEDULE_URL
        )
        response_status_code = (
            response_status_value
            if type(response_status_value) is int
            else 0
        )
        response_redirect_count = (
            len(response_history)
            if isinstance(response_history, (list, tuple))
            else 0
        )

    response_headers = getattr(response, "headers", {})
    http_date_value = (
        response_headers.get("Date")
        if hasattr(response_headers, "get")
        else None
    )
    mlb_http_date_utc = (
        _parse_http_date(http_date_value)
        if require_http_date
        else None
    )
    mlb_http_date_header_raw = (
        http_date_value
        if require_http_date and isinstance(http_date_value, str)
        else None
    )

    if require_http_date and mlb_http_date_utc is not None:
        _validate_observed_clock(
            mlb_http_date_utc=mlb_http_date_utc,
            response_received_at=response_received_at,
        )

    return _ScheduleHTTPResponse(
        raw_content=raw_content,
        payload=payload,
        response_effective_url=response_effective_url,
        response_status_code=response_status_code,
        response_redirect_count=response_redirect_count,
        mlb_http_date_header_raw=mlb_http_date_header_raw,
        mlb_http_date_utc=mlb_http_date_utc,
        response_received_at_utc=response_received_at_utc,
        response_body_sha256=response_body_sha256,
    )


def _request_schedule(
    parameters: dict[str, object],
) -> tuple[bytes, dict[str, Any]]:
    """Télécharge puis décode une réponse MLB unique."""
    observation = _request_schedule_response(
        parameters,
        require_http_date=False,
    )
    return observation.raw_content, observation.payload


def _normalize_game_types(
    game_types: Iterable[str],
) -> tuple[str, ...]:
    """Normalise les types de matchs demandés."""
    normalized_types = tuple(
        sorted(
            {
                str(game_type)
                .strip()
                .upper()
                for game_type in game_types
                if str(game_type).strip()
            }
        )
    )

    if not normalized_types:
        raise ValueError(
            "Au moins un type de match "
            "est obligatoire."
        )

    return normalized_types


def fetch_schedule(
    target_date: date,
) -> list[ScheduledGame]:
    """Récupère les matchs MLB d’une journée pour l’interface."""
    parameters: dict[str, object] = {
        "sportId": MLB_SPORT_ID,
        "date": target_date.isoformat(),
        "hydrate": "probablePitcher",
    }

    _, payload = _request_schedule(
        parameters
    )

    return _parse_schedule_payload(
        payload
    )


def fetch_schedule_range(
    start_date: date,
    end_date: date,
    game_types: Iterable[str] = ("R",),
) -> ScheduleFetchResult:
    """Récupère une période auditable limitée à 31 jours."""
    if end_date < start_date:
        raise ValueError(
            "La date de fin ne peut pas "
            "précéder la date de début."
        )

    inclusive_days = (
        end_date - start_date
    ).days + 1

    if (
        inclusive_days
        > MAX_SCHEDULE_RANGE_DAYS
    ):
        raise ValueError(
            "Une récupération ne peut pas "
            "dépasser 31 jours."
        )

    normalized_game_types = (
        _normalize_game_types(
            game_types
        )
    )

    parameters: dict[str, object] = {
        "sportId": MLB_SPORT_ID,
        "startDate": (
            start_date.isoformat()
        ),
        "endDate": (
            end_date.isoformat()
        ),
        "gameTypes": ",".join(
            normalized_game_types
        ),
        "hydrate": "probablePitcher",
    }

    (
        raw_content,
        payload,
    ) = _request_schedule(
        parameters
    )

    games = _parse_schedule_payload(
        payload
    )

    return ScheduleFetchResult(
        start_date=start_date,
        end_date=end_date,
        game_types=normalized_game_types,
        request_parameters=dict(
            parameters
        ),
        raw_content=raw_content,
        games=tuple(games),
    )


def fetch_schedule_range_observed(
    start_date: date,
    end_date: date,
    game_types: Iterable[str] = ("R",),
) -> ScheduleFetchResult:
    """Récupère une période avec les preuves HTTP nécessaires au shadow."""
    if end_date < start_date:
        raise ValueError(
            "La date de fin ne peut pas précéder la date de début."
        )

    inclusive_days = (end_date - start_date).days + 1
    if inclusive_days > MAX_SCHEDULE_RANGE_DAYS:
        raise ValueError(
            "Une récupération ne peut pas dépasser 31 jours."
        )

    normalized_game_types = _normalize_game_types(game_types)
    parameters: dict[str, object] = {
        "sportId": MLB_SPORT_ID,
        "startDate": start_date.isoformat(),
        "endDate": end_date.isoformat(),
        "gameTypes": ",".join(normalized_game_types),
        "hydrate": "probablePitcher",
    }

    observation = _request_schedule_response(
        parameters,
        require_http_date=True,
    )
    games = _parse_schedule_payload(
        observation.payload,
        reject_duplicate_game_ids=True,
    )

    return ScheduleFetchResult(
        start_date=start_date,
        end_date=end_date,
        game_types=normalized_game_types,
        request_parameters=dict(parameters),
        raw_content=observation.raw_content,
        games=tuple(games),
        response_effective_url=observation.response_effective_url,
        response_status_code=observation.response_status_code,
        response_redirect_count=observation.response_redirect_count,
        mlb_http_date_header_raw=(
            observation.mlb_http_date_header_raw
        ),
        mlb_http_date_utc=observation.mlb_http_date_utc,
        mlb_http_response_received_at_utc=(
            observation.response_received_at_utc
        ),
        response_body_sha256=observation.response_body_sha256,
    )


def _parse_date(
    value: str,
) -> date:
    """Valide une date écrite sous la forme AAAA-MM-JJ."""
    try:
        return date.fromisoformat(
            value
        )

    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "La date doit respecter "
            "le format AAAA-MM-JJ."
        ) from error


def _pitcher_label(
    pitcher_name: str | None,
) -> str:
    """Prépare un nom de lanceur lisible."""
    return (
        pitcher_name
        or "non annoncé"
    )


def _display_games(
    games: Sequence[ScheduledGame],
) -> None:
    """Affiche un échantillon lisible des matchs."""
    for game in games[:DISPLAY_LIMIT]:
        away_pitcher = _pitcher_label(
            game.away_probable_pitcher_name
        )

        home_pitcher = _pitcher_label(
            game.home_probable_pitcher_name
        )

        print(
            f"- {game.away_team_name} "
            f"({away_pitcher})"
            f" @ {game.home_team_name} "
            f"({home_pitcher})"
            f" | {game.status_detail}"
        )

    hidden_games = (
        len(games)
        - DISPLAY_LIMIT
    )

    if hidden_games > 0:
        print(
            f"... {hidden_games} "
            "autres matchs non affichés"
        )


def main() -> None:
    """Teste une journée ou une période depuis le terminal."""
    parser = argparse.ArgumentParser(
        description=(
            "Récupère le calendrier MLB."
        )
    )

    parser.add_argument(
        "--date",
        dest="target_date",
        type=_parse_date,
        help=(
            "Journée au format AAAA-MM-JJ."
        ),
    )

    parser.add_argument(
        "--start-date",
        dest="start_date",
        type=_parse_date,
        help=(
            "Début de période au format "
            "AAAA-MM-JJ."
        ),
    )

    parser.add_argument(
        "--end-date",
        dest="end_date",
        type=_parse_date,
        help=(
            "Fin de période au format "
            "AAAA-MM-JJ."
        ),
    )

    arguments = parser.parse_args()

    has_start_date = (
        arguments.start_date
        is not None
    )

    has_end_date = (
        arguments.end_date
        is not None
    )

    if (
        has_start_date
        != has_end_date
    ):
        parser.error(
            "--start-date et --end-date "
            "doivent être utilisés ensemble."
        )

    if (
        arguments.target_date is not None
        and has_start_date
    ):
        parser.error(
            "--date ne peut pas être "
            "combiné avec une période."
        )

    try:
        if (
            has_start_date
            and has_end_date
        ):
            result = fetch_schedule_range(
                arguments.start_date,
                arguments.end_date,
            )

            games: Sequence[
                ScheduledGame
            ] = result.games

            print(
                "Période demandée : "
                f"{result.start_date.isoformat()} "
                "au "
                f"{result.end_date.isoformat()}"
            )

            print(
                "Octets bruts reçus : "
                f"{len(result.raw_content)}"
            )

        else:
            target_date = (
                arguments.target_date
                or date.today()
            )

            games = fetch_schedule(
                target_date
            )

            print(
                "Date demandée : "
                f"{target_date.isoformat()}"
            )

    except (
        MLBAPIError,
        ValueError,
    ) as error:
        raise SystemExit(
            f"Erreur MLB : {error}"
        ) from error

    print(
        "Matchs MLB récupérés : "
        f"{len(games)}"
    )

    if not games:
        print(
            "Aucun match MLB trouvé."
        )
        return

    _display_games(games)


if __name__ == "__main__":
    main()
