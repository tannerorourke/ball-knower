"""Injury and IR flags for my roster: hurt starters, free reserve slots, reserve players cleared to play.

Reserve eligibility follows the league's own `reserve_allow_*` settings. `after_reserve_moves` applies
the reserve flags to my roster so lineup and waiver scoring see the roster as it will stand.

    python3 -m sleeper_agent injuries
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from ..core.league import LeagueSnapshot, TeamRoster, open_league

STARTER_STATUSES = {"Questionable", "Doubtful", "Out", "IR"}
RESERVE_MOVES = ("ir_open", "ir_cleared")


@dataclass
class Flag:
    kind: str  # -- starter_hurt | ir_open | ir_cleared
    player_id: str
    text: str


def _label(pid: str, players: dict) -> str:
    m = players.get(pid) or {}
    return f"{m.get('name') or pid} ({m.get('pos') or '?'}, {m.get('team') or 'FA'})"


def injury_flags(snap: LeagueSnapshot, team: TeamRoster, players: dict) -> list[Flag]:
    """At most one `ir_open` flag per free reserve slot, in roster order."""
    flags = []
    status = lambda pid: (players.get(pid) or {}).get("injury_status")
    for pid in team.starters:
        if status(pid) in STARTER_STATUSES:
            flags.append(Flag("starter_hurt", pid, f"STARTER {_label(pid, players)} is {status(pid)}"))
    free = snap.reserve_slots - len(team.reserve)
    placed = set(team.starters) | set(team.reserve) | set(team.taxi)
    for pid in team.players:
        if free <= 0:
            break
        if pid not in placed and status(pid) in snap.reserve_statuses:
            flags.append(Flag("ir_open", pid, f"BENCH {_label(pid, players)} is {status(pid)}; {free} reserve slot(s) free"))
            free -= 1
    for pid in team.reserve:
        if not status(pid):
            flags.append(Flag("ir_cleared", pid, f"RESERVE {_label(pid, players)} has no injury tag; move off reserve"))
    return flags


def after_reserve_moves(snap: LeagueSnapshot, players: dict) -> tuple[LeagueSnapshot, list[Flag]]:
    """-> (the snapshot after the recommended reserve moves, those flags): `ir_open` players count on reserve,
    `ir_cleared` players back in play."""
    moves = [f for f in injury_flags(snap, snap.mine, players) if f.kind in RESERVE_MOVES]
    if not moves:
        return snap, []
    me = snap.mine
    cleared = {f.player_id for f in moves if f.kind == "ir_cleared"}
    reserve = [p for p in me.reserve if p not in cleared] + [f.player_id for f in moves if f.kind == "ir_open"]
    moved = replace(me, reserve=reserve)
    return replace(snap, rosters=[moved if r.is_mine else r for r in snap.rosters]), moves


def describe_move(f: Flag, players: dict) -> str:
    return f"{_label(f.player_id, players)} {'to' if f.kind == 'ir_open' else 'off'} reserve"


def format_flags(flags: list[Flag]) -> str:
    return "\n".join(f.text for f in flags) if flags else "no injury flags"


# -- 'week' is unused: Sleeper keeps only the current injury tags
def flags_for_week(api, snap: LeagueSnapshot, cfg: dict, week: int) -> list[Flag]:
    return injury_flags(snap, snap.mine, api.players())


def cmd_injuries(cfg, args):
    api, snap = open_league(cfg)
    print(format_flags(flags_for_week(api, snap, cfg, snap.week)))
