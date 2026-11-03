# Verified NHL team data collection and reuse

The team CSV is collected separately from the match calculation. A slow CSV
transfer must not hold the odds capture workflow's concurrency lock.

The dedicated private `NHL verified team data refresh` workflow permits two
15-minute transfer attempts, with a 35-minute job ceiling. The same public
MoneyPuck CSV and gzip/length/integrity checks are used. Only a fully parsed,
archived and reverified capture can be saved to the private repository's Actions
cache. Cache eviction is possible: a missing cache requires a new successful
collection, never a fabricated or historical fallback.

The collector explicitly refreshes at 10:15, 13:15, 16:15 and 19:15 Paris time
in the NHL season months. On a successful default-branch collection, the
existing private research calculation is triggered automatically. Calculation
can also be launched manually after collection finishes.

Consumers restore the latest available current-Paris-day cache, check hashes,
protocol and original receipt time, and copy only the newest verified capture
into their engine data directory. Captures must be automatic, no more than six
hours old, and from the same Paris day. Future-dated, altered, incomplete and
manual captures are rejected. Restore time never replaces source receipt time.
The existing per-match H-1 cutoff and the existing statistical model remain in
force. A refresh after a game's cutoff cannot backfill a pregame observation.

`python -m nhl_operations.team_data_cache collect --force-refresh --code-commit SHA --root PATH`
collects fresh evidence; without `--force-refresh` it may reuse eligible local
evidence. `check` verifies it without a request. `materialize --destination PATH`
verifies and copies only the selected slot, without replacing an existing slot.

The three calculation workflows must call `automatic_research_capture` with
`--cached-team-only`. This fetches a new official schedule but never downloads
the team CSV. A missing/stale cache produces an explicit error directing the
operator to the collection workflow. Full live collection and end-to-end model
output must still be verified after activation; offline tests alone do not
prove the source will finish downloading within the larger time allowance.
