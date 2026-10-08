"""Manual NHL historical-final pilot: review a plan before at most ten GETs."""
from __future__ import annotations

import argparse
from pathlib import Path

from src.nhl.direct_final_transport import curl_direct_nhl_json
from src.nhl.historical_final_pilot import (
    capture_historical_final_pilot, plan_historical_final_pilot,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Lot manuel de finales NHL historiques")
    parser.add_argument("team_slot", type=Path, help="capture MoneyPuck deja verifiee")
    parser.add_argument("--limit", type=int, default=10, help="maximum de 1 a 10 GET")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan", action="store_true", help="aucun appel reseau")
    mode.add_argument("--capture", action="store_true", help="executer le plan revu")
    parser.add_argument("--expected-backlog-sha256", metavar="SHA256")
    args = parser.parse_args()
    if args.plan:
        if args.expected_backlog_sha256 is not None:
            parser.error("L'empreinte n'est utilisee qu'avec --capture.")
        plan = plan_historical_final_pilot(args.team_slot, request_limit=args.limit)
        print("Source SHA-256:", plan.source_capture_sha256)
        print("Retard SHA-256:", plan.backlog_sha256)
        print("Matchs a controler:", ", ".join(map(str, plan.game_ids)) or "aucun")
        print("GET maximum:", len(plan.game_ids))
        print("Entrainement: NON ; publication de pronostics: NON")
        return
    if args.expected_backlog_sha256 is None:
        parser.error("--capture exige --expected-backlog-sha256.")
    receipts = capture_historical_final_pilot(
        args.team_slot, expected_backlog_sha256=args.expected_backlog_sha256,
        explicit_manual_run=True, request_limit=args.limit,
        transport=curl_direct_nhl_json,
    )
    for receipt in receipts:
        print(receipt.game_id, receipt.final_type, receipt.winner_abbr, receipt.path)
    print("Captures reussies:", len(receipts))
    print("Entrainement: NON ; publication de pronostics: NON")


if __name__ == "__main__":
    main()
