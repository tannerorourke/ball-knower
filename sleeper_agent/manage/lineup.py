"""Best starting lineup for a week, and the swaps from the current one.

Fixed slots take the top projected players at their position, then flex slots fill from most to
least restrictive. Reserve and taxi players, counted after the recommended reserve moves, are out
of play.

    python3 -m sleeper_agent lineup [--week N]
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from ..league import LeagueSnapshot, open_league
from ..scoring import Slots
from .injuries import after_reserve_moves, describe_move
from .projections import WeekProjection, project_week


@dataclass
class Swap:
    start: WeekProjection
    sit: WeekProjection | None
    delta: float


def best_lineup(candidates: list[WeekProjection], slots: Slots) -> list[WeekProjection]:
    """Fixed slots take their position's top players; flex slots, most restrictive first, take the best left."""
    chosen: list[WeekProjection] = []
    taken: set[str] = set()

    def take(eligible: set[str], n: int) -> None:
        pool = sorted((c for c in candidates if c.pos in eligible and c.player_id not in taken),
                      key=lambda c: (-c.pts, c.player_id))
        for c in pool[:n]:
            chosen.append(c)
            taken.add(c.player_id)

    for pos, n in slots.fixed.items():
        if n:
            take({pos}, n)
    for pos, n in Counter(slots.ignored).items():     # -- DEF and any other slot a player position fills
        take({pos}, n)
    for elig in sorted(slots.flex, key=len):
        take(elig, 1)
    return chosen


def diff_lineups(current: list[WeekProjection], best: list[WeekProjection]) -> list[Swap]:
    cur_ids, best_ids = {c.player_id for c in current}, {b.player_id for b in best}
    starts = sorted((b for b in best if b.player_id not in cur_ids), key=lambda p: -p.pts)
    sits = sorted((c for c in current if c.player_id not in best_ids), key=lambda p: p.pts)
    swaps = []
    for s in starts:
        match = next((x for x in sits if x.pos == s.pos), None) or (sits[0] if sits else None)
        if match is not None:
            sits.remove(match)
        swaps.append(Swap(s, match, s.pts - (match.pts if match else 0.0)))
    return swaps


def recommend(snap: LeagueSnapshot, proj: dict[str, WeekProjection]) -> tuple[list[WeekProjection], list[WeekProjection], list[Swap]]:
    """-> (best lineup, current modeled starters, swaps) for my roster."""
    me = snap.mine
    slots = Slots.from_roster_positions(snap.roster_positions)
    out_of_play = set(me.reserve) | set(me.taxi)
    pool = [proj[p] for p in me.players if p in proj and p not in out_of_play]
    best = best_lineup(pool, slots)
    current = [proj[p] for p in me.starters if p in proj]
    return best, current, diff_lineups(current, best)


def with_defenses(players: dict, snap: LeagueSnapshot) -> dict:
    """Add team-defense entries for rostered ids the player table lacks; a defense's id is its team."""
    slots = Slots.from_roster_positions(snap.roster_positions)
    if "DEF" not in slots.ignored:
        return players
    out = dict(players)
    for pid in snap.mine.players:
        if pid not in out and pid.isalpha() and pid.isupper():
            out[pid] = {"player_id": pid, "name": f"{pid} DEF", "pos": "DEF", "team": pid, "injury_status": None}
    return out


def _tag(p: WeekProjection) -> str:
    return f" [{p.note}]" if p.note else ""


def format_report(week: int, best, current, swaps: list[Swap]) -> str:
    lines = [f"Week {week} lineup: current {sum(c.pts for c in current):.1f} pts, "
             f"best {sum(b.pts for b in best):.1f} pts"]
    if best:
        lines.append("Starting slots: " + ", ".join(f"{b.pos} {b.name}" for b in best))
    if not swaps:
        lines.append("Current starters are already optimal.")
    for s in swaps:
        sit = f"SIT   {s.sit.name} ({s.sit.pos}) {s.sit.pts:.1f}{_tag(s.sit)}" if s.sit else "SIT   (empty slot)"
        lines.append(f"START {s.start.name} ({s.start.pos}) {s.start.pts:.1f}{_tag(s.start)}  |  {sit}  |  {s.delta:+.1f}")
    return "\n".join(lines)


@dataclass
class WeekLineup:
    best: list[WeekProjection]
    current: list[WeekProjection]
    swaps: list[Swap]
    proj: dict[str, WeekProjection]
    moves: list[str]


def lineup_for_week(api, snap: LeagueSnapshot, cfg: dict, week: int) -> WeekLineup:
    byes = api.byes(snap.season)
    bye_teams = {t for t, w in byes.items() if w == week}
    extra = {"DEF"} & set(snap.roster_positions)
    players = with_defenses(api.players(), snap)
    moved, moves = after_reserve_moves(snap, players)
    proj = project_week(api.week_projections(snap.season, week, extra), players, snap.scoring_settings,
                        bye_teams, cfg.get("inseason"), week == snap.week)
    return WeekLineup(*recommend(moved, proj), proj, [describe_move(f, players) for f in moves])


def cmd_lineup(cfg, args):
    api, snap = open_league(cfg)
    week = args.week or snap.week
    r = lineup_for_week(api, snap, cfg, week)
    print(format_report(week, r.best, r.current, r.swaps))
    if r.moves:
        print("Assumes the reserve moves: " + "; ".join(r.moves))
