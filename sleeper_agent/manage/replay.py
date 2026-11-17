"""Replay a past week: the league's rosters and FAAB rebuilt to a cutoff by undoing every later transaction,
read through an API wrapper that hides what was not known at that cutoff.

    python3 -m sleeper_agent manage --as-of-week N [--no-llm]
"""
from __future__ import annotations

import datetime
import sys
from dataclasses import replace

from ..league import LeagueSnapshot, open_league

LAST_ROUND = 18


def cutoff_for_week(schedule: list, week: int) -> int:
    """12:00 UTC on the day after the last game of week `week - 1`, in epoch milliseconds."""
    dates = [str(g["date"]) for g in schedule if int(g.get("week") or 0) == week - 1 and g.get("date")]
    if not dates:
        raise ValueError(f"no games in week {week - 1} to cut off after")
    day = datetime.date.fromisoformat(max(dates)) + datetime.timedelta(days=1)
    noon = datetime.datetime(day.year, day.month, day.day, 12, tzinfo=datetime.timezone.utc)
    return int(noon.timestamp() * 1000)


def cutoff_label(cutoff_ms: int) -> str:
    return datetime.datetime.fromtimestamp(cutoff_ms / 1000, datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _stamp(t: dict) -> int:
    return int(t.get("status_updated") or 0)


# -- each complete transaction after the cutoff undone, newest first, bids refunded; starters and reserve
#    come back empty, since Sleeper keeps no past lineup
def snapshot_as_of(snap: LeagueSnapshot, transactions: list, cutoff_ms: int) -> LeagueSnapshot:
    teams = {r.roster_id: replace(r, players=list(r.players), starters=[], reserve=[], taxi=list(r.taxi))
             for r in snap.rosters}
    later = sorted((t for t in transactions if t.get("status") == "complete" and _stamp(t) > cutoff_ms),
                   key=_stamp, reverse=True)
    for t in later:
        adds, drops = t.get("adds") or {}, t.get("drops") or {}
        for pid, rid in adds.items():
            team = teams.get(int(rid))
            if team:
                team.players = [p for p in team.players if p != pid]
                team.taxi = [p for p in team.taxi if p != pid]
        for pid, rid in drops.items():
            team = teams.get(int(rid))
            if team and pid not in team.players:
                team.players.append(pid)
        bid = (t.get("settings") or {}).get("waiver_bid")
        if t.get("type") == "waiver" and bid:
            for rid in set(adds.values()):
                team = teams.get(int(rid))
                if team:
                    team.faab_used -= float(bid)
                    team.faab_left += float(bid)
    return replace(snap, rosters=[teams[r.roster_id] for r in snap.rosters])


class ReplayApi:
    """Sleeper as of the replayed week: injury tags blank, each player on the team of his projection row for
    that week, only transactions up to the cutoff, no trending players. Every other call passes through."""

    def __init__(self, api, snap: LeagueSnapshot, week: int, cutoff_ms: int, rounds: dict[int, list]):
        self._api, self._week, self._cutoff, self._rounds = api, week, cutoff_ms, rounds
        self._season = snap.season
        self._extra = {"DEF"} & set(snap.roster_positions)
        self._proj: dict[tuple, list] = {}
        self._table: dict | None = None

    def __getattr__(self, name):
        return getattr(self._api, name)

    def week_projections(self, season: int, week: int, extra_positions=()) -> list:
        key = (int(season), int(week), frozenset(extra_positions))
        if key not in self._proj:
            self._proj[key] = self._api.week_projections(season, week, extra_positions)
        return self._proj[key]

    def players(self, *args, **kwargs) -> dict:
        if self._table is None:
            rows = self.week_projections(self._season, self._week, self._extra)
            teams = {str(r.get("player_id")): r["team"] for r in rows if r.get("team")}
            self._table = {pid: {**m, "injury_status": None, "team": teams.get(pid, m.get("team"))}
                           for pid, m in self._api.players(*args, **kwargs).items()}
        return self._table

    def trending(self, *args, **kwargs) -> list:
        return []

    def transactions(self, league_id: str, round_: int) -> list:
        return [t for t in self._rounds.get(int(round_), []) if _stamp(t) <= self._cutoff]


def open_replay(cfg: dict, week: int) -> tuple[ReplayApi, LeagueSnapshot, dict]:
    """-> (the replay API, the league as of the cutoff before `week`, the replay block for the packet)."""
    api, snap = open_league(cfg)
    try:
        cutoff = cutoff_for_week(api.schedule(snap.season), week)
    except ValueError as e:
        sys.exit(f"--as-of-week {week}: {e}")
    lid = str(cfg["league_id"])
    rounds = {r: api.transactions(lid, r) for r in range(1, max(LAST_ROUND, snap.week) + 1)}
    past = snapshot_as_of(snap, [t for ts in rounds.values() for t in ts], cutoff)
    info = {"as_of_week": week, "cutoff": cutoff_label(cutoff)}
    return ReplayApi(api, past, week, cutoff, rounds), replace(past, week=week), info
