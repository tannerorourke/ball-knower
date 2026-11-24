"""Weekly point projections with bye and injury adjustments.

Sleeper's per-week stat lines are scored under the league's scoring settings. A bye or an out-type
injury tag projects 0; Questionable and Doubtful scale by `inseason.*_discount`. A game-status tag
describes one game, so it adjusts only the current NFL week. A library for `lineup` and `waivers`.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..core.scoring import score_stats

OUT_STATUSES = {"Out", "IR", "PUP", "Sus", "COV", "NA", "DNR"}
GAME_STATUSES = {"Questionable", "Doubtful", "Out"}
DEFAULT_DISCOUNTS = {"questionable_discount": 0.85, "doubtful_discount": 0.4}


@dataclass
class WeekProjection:
    player_id: str
    name: str
    pos: str
    team: str | None
    raw: float
    pts: float
    status: str | None
    note: str = ""


def adjusted_points(raw: float, status: str | None, on_bye: bool, inseason: dict | None = None,
                    current: bool = True) -> tuple[float, str]:
    """-> (points, note). Bye and an out-type tag score 0; Questionable and Doubtful are discounted.
    Outside the `current` week a game-status tag leaves the raw projection standing."""
    cfg = {**DEFAULT_DISCOUNTS, **(inseason or {})}
    if on_bye:
        return 0.0, "bye"
    if status in GAME_STATUSES and not current:
        return raw, ""
    if status in OUT_STATUSES:
        return 0.0, "out"
    if status == "Doubtful":
        return raw * float(cfg["doubtful_discount"]), "doubtful"
    if status == "Questionable":
        return raw * float(cfg["questionable_discount"]), "questionable"
    return raw, ""


# -- scores Sleeper's per-week stat lines; a player in 'players' with no line projects 0.
#    'current' is whether this is the current NFL week, the only week a game-status tag adjusts
def project_week(raw_projections: list, players: dict, scoring: dict, bye_teams: set[str] | frozenset = frozenset(),
                 inseason: dict | None = None, current: bool = True) -> dict[str, WeekProjection]:
    lines = {str(p.get("player_id")): (p.get("stats") or {}) for p in raw_projections}
    out = {}
    for pid, meta in players.items():
        pid = str(pid)
        team = meta.get("team")
        raw = score_stats(lines.get(pid, {}), scoring)
        status = meta.get("injury_status")
        pts, note = adjusted_points(raw, status, team in bye_teams, inseason, current)
        out[pid] = WeekProjection(pid, meta.get("name") or pid, meta.get("pos") or "", team,
                                  round(raw, 2), round(pts, 2), status, note)
    return out
