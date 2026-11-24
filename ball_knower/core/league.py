"""In-season league snapshot: rosters, owners, current week and FAAB. Needs no draft.

Built from Sleeper's league, users, rosters and NFL-state endpoints. Also resolves which injury
statuses the league lets onto reserve; every weekly command in `manage` starts from it.

    python3 -m ball_knower league
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field

RESERVE_FLAGS = {"Out": "reserve_allow_out", "Doubtful": "reserve_allow_doubtful", "Sus": "reserve_allow_sus",
                 "COV": "reserve_allow_cov", "DNR": "reserve_allow_dnr", "NA": "reserve_allow_na"}


@dataclass
class TeamRoster:
    roster_id: int
    owner_id: str | None
    owner: str
    players: list[str]
    starters: list[str]
    reserve: list[str]
    taxi: list[str]
    wins: int
    losses: int
    ties: int
    faab_used: float
    faab_left: float
    waiver_position: int | None
    is_mine: bool = False


@dataclass
class LeagueSnapshot:
    league_id: str
    name: str
    season: int
    week: int
    season_type: str
    roster_positions: list[str]
    scoring_settings: dict
    faab_budget: float
    rosters: list[TeamRoster] = field(default_factory=list)
    reserve_slots: int = 0
    reserve_statuses: frozenset = frozenset({"IR"})

    @property
    def mine(self) -> TeamRoster | None:
        return next((r for r in self.rosters if r.is_mine), None)

    @classmethod
    def from_api(cls, api, league_id: str, username: str | None = None) -> "LeagueSnapshot":
        league = api.league(league_id)
        users = api.league_users(league_id)
        rosters = api.league_rosters(league_id)
        state = api.nfl_state()
        my_uid = str(api.user(username)["user_id"]) if username else None
        return cls.build(league, users, rosters, state, my_uid)

    @classmethod
    def build(cls, league: dict, users: list, rosters: list, state: dict, my_user_id: str | None = None):
        names = {str(u["user_id"]): (u.get("display_name") or str(u["user_id"])) for u in users}
        settings = league.get("settings") or {}
        budget = float(settings.get("waiver_budget") or 0)
        allowed = {"IR"} | {st for st, key in RESERVE_FLAGS.items() if settings.get(key)}
        out = []
        for r in rosters:
            rs = r.get("settings") or {}
            owner = str(r["owner_id"]) if r.get("owner_id") else None
            used = float(rs.get("waiver_budget_used") or 0)
            out.append(TeamRoster(
                roster_id=int(r["roster_id"]), owner_id=owner,
                owner=names.get(owner, owner or f"roster {r['roster_id']}"),
                players=list(r.get("players") or []), starters=list(r.get("starters") or []),
                reserve=list(r.get("reserve") or []), taxi=list(r.get("taxi") or []),
                wins=int(rs.get("wins") or 0), losses=int(rs.get("losses") or 0), ties=int(rs.get("ties") or 0),
                faab_used=used, faab_left=budget - used,
                waiver_position=rs.get("waiver_position"),
                is_mine=bool(my_user_id and owner == str(my_user_id)),
            ))
        return cls(
            league_id=str(league["league_id"]), name=league.get("name") or "", season=int(league.get("season") or state.get("season") or 0),
            week=int(state.get("week") or 0), season_type=state.get("season_type") or "regular",
            roster_positions=list(league.get("roster_positions") or []),
            scoring_settings=dict(league.get("scoring_settings") or {}),
            faab_budget=budget, rosters=out,
            reserve_slots=int(settings.get("reserve_slots") or 0), reserve_statuses=frozenset(allowed),
        )


def open_league(cfg: dict):
    """-> (api, snapshot) for the configured league; exits with one line when config or roster is missing."""
    from .sleeper import Sleeper
    for key in ("league_id", "username"):
        if not cfg.get(key):
            sys.exit(f"config key {key} is not set")
    api = Sleeper(cfg["data_dir"])
    snap = LeagueSnapshot.from_api(api, str(cfg["league_id"]), cfg["username"])
    if snap.mine is None:
        sys.exit(f"no roster in this league belongs to {cfg['username']}")
    return api, snap
