"""Injury and IR flags for my roster: hurt starters, free reserve slots, reserve players cleared to play.

Reserve eligibility follows the league's own `reserve_allow_*` settings.

    python3 -m sleeper_agent injuries
"""
from __future__ import annotations

from dataclasses import dataclass

from ..league import LeagueSnapshot, TeamRoster, open_league

STARTER_STATUSES = {"Questionable", "Doubtful", "Out", "IR"}


@dataclass
class Flag:
    kind: str  # -- starter_hurt | ir_open | ir_cleared
    player_id: str
    text: str


def _label(pid: str, players: dict) -> str:
    m = players.get(pid) or {}
    return f"{m.get('name') or pid} ({m.get('pos') or '?'}, {m.get('team') or 'FA'})"


def injury_flags(snap: LeagueSnapshot, team: TeamRoster, players: dict) -> list[Flag]:
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
    for pid in team.reserve:
        if not status(pid):
            flags.append(Flag("ir_cleared", pid, f"RESERVE {_label(pid, players)} has no injury tag; move off reserve"))
    return flags


def format_flags(flags: list[Flag]) -> str:
    return "\n".join(f.text for f in flags) if flags else "no injury flags"


def cmd_injuries(cfg, args):
    api, snap = open_league(cfg)
    print(format_flags(injury_flags(snap, snap.mine, api.players())))
