"""Waiver pairs: add a free agent, drop a rostered player, ranked into a CSV by lineup gain.

Gain is scored on my roster after the recommended reserve moves. Each add carries a FAAB band from
past winning bids; each drop carries `ros`, its projection through week 17, which breaks gain ties.

    python3 -m ball_knower waivers [--week N]
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass

from ..core.league import LeagueSnapshot, open_league
from ..core.scoring import Slots
from .injuries import after_reserve_moves, describe_move
from .lineup import best_lineup, with_defenses
from .projections import WeekProjection, project_week

DEFAULT_POOL = 40
DEFAULT_HORIZON = 3
AUTO_GROUPS = 8
AUTO_TRENDING = 5
ALT_WINDOW = 0.1
MAX_ALTERNATIVES = 3
ROS_LAST_WEEK = 17


@dataclass
class Pair:
    add: WeekProjection
    drop: WeekProjection | None
    gain: float
    add_total: float
    trending_adds: int
    faab_low: float | None
    faab_high: float | None
    drop_total: float = 0.0
    drop_ros: float = 0.0


def percentile(values: list[float], q: float) -> float:
    """Linear-interpolated percentile, q in [0, 100]."""
    xs = sorted(values)
    k = (len(xs) - 1) * q / 100
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def winning_bids(transactions: list, players: dict) -> dict[str, list[float]]:
    """Completed waiver bids by the position of the player won."""
    out: dict[str, list[float]] = {}
    for t in transactions:
        if t.get("type") != "waiver" or t.get("status") != "complete":
            continue
        bid = (t.get("settings") or {}).get("waiver_bid")
        if bid is None:
            continue
        for pid in (t.get("adds") or {}):
            pos = (players.get(pid) or {}).get("pos")
            if pos:
                out.setdefault(pos, []).append(float(bid))
    return out


def faab_band(bids: dict[str, list[float]], pos: str) -> tuple[float, float] | None:
    """25th to 75th percentile of the league's past winning bids at a position; None without any."""
    xs = bids.get(pos)
    if not xs:
        return None
    return round(percentile(xs, 25)), round(percentile(xs, 75))


def free_agent_table(players: dict, snap: LeagueSnapshot) -> dict:
    """The player table plus a defense for every team in it (when the league starts one)."""
    out = with_defenses(players, snap)
    if "DEF" not in Slots.from_roster_positions(snap.roster_positions).ignored:
        return out
    out = dict(out)
    for team in {m.get("team") for m in players.values() if m.get("team")}:
        out.setdefault(team, {"player_id": team, "name": f"{team} DEF", "pos": "DEF", "team": team,
                              "injury_status": None})
    return out


def _lineup_total(pool: list[WeekProjection], slots: Slots) -> float:
    return sum(c.pts for c in best_lineup(pool, slots))


def horizon_total(weekly: list[dict[str, WeekProjection]], pid: str) -> float:
    return sum(w[pid].pts for w in weekly if pid in w)


# -- projection summed over 'weekly' for every player in it; 'weekly' runs from the report week to the season's end
def rest_of_season(weekly: list[dict[str, WeekProjection]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for w in weekly:
        for pid, p in w.items():
            out[pid] = out.get(pid, 0.0) + p.pts
    return {pid: round(v, 2) for pid, v in out.items()}


def ros_week_count(week: int) -> int:
    return max(week, ROS_LAST_WEEK) - week + 1


def started_positions(roster_positions: list[str]) -> set[str]:
    slots = Slots.from_roster_positions(roster_positions)
    return {p for p, n in slots.fixed.items() if n} | set().union(*slots.flex) | set(slots.ignored)


def rostered_ids(snap: LeagueSnapshot) -> set[str]:
    return {p for r in snap.rosters for p in (*r.players, *r.reserve, *r.taxi)}


# -- the top 'pool_size' free agents (players on no roster) by projection summed over 'weekly'
def candidate_ids(snap: LeagueSnapshot, weekly: list[dict[str, WeekProjection]],
                  pool_size: int = DEFAULT_POOL) -> list[str]:
    rostered = rostered_ids(snap)
    ids = set().union(*(w.keys() for w in weekly)) if weekly else set()
    return sorted((p for p in ids if p not in rostered and horizon_total(weekly, p) > 0),
                  key=lambda p: (-horizon_total(weekly, p), p))[:pool_size]


def weekly_projections(api, snap: LeagueSnapshot, players: dict, week: int, horizon: int,
                       inseason: dict) -> list[dict[str, WeekProjection]]:
    """One projection dict per week of the horizon, starting at `week`; `players` is the free-agent table.
    Game-status tags adjust only `snap.week`, the current NFL week."""
    byes = api.byes(snap.season)
    extra = {"DEF"} & set(snap.roster_positions)
    weekly = []
    for w in range(week, week + horizon):
        bye_teams = {t for t, bw in byes.items() if bw == w}
        weekly.append(project_week(api.week_projections(snap.season, w, extra), players, snap.scoring_settings,
                                   bye_teams, inseason, w == snap.week))
    return weekly


# -- trending add counts by player id, and the league's past winning bids by position
def market_inputs(api, league_id: str, week: int, players: dict) -> tuple[dict[str, int], dict[str, list[float]]]:
    trending = {str(t.get("player_id")): int(t.get("count") or 0) for t in api.trending("add")}
    past = [t for r in range(1, week + 1) for t in api.transactions(league_id, r)]
    return trending, winning_bids(past, players)


def trending_free_agents(snap: LeagueSnapshot, trending: dict[str, int], weekly: list[dict[str, WeekProjection]],
                         limit: int | None = None) -> list[str]:
    """Trending players on no roster, most added first; all of them when `limit` is None. Only ids the
    player table (`weekly[0]`) holds at a position the league starts are kept."""
    table = weekly[0] if weekly else {}
    started = started_positions(snap.roster_positions)
    rostered = rostered_ids(snap)
    ids = sorted((p for p in trending if p not in rostered and p in table and table[p].pos in started),
                 key=lambda p: (-trending[p], p))
    return ids if limit is None else ids[:limit]


# -- the adds of the best 'groups' add groups by gain, then the 'trending_n' most-added free agents
def auto_candidates(snap: LeagueSnapshot, weekly: list[dict[str, WeekProjection]], trending: dict[str, int],
                    bids: dict[str, list[float]], pool_size: int = DEFAULT_POOL, groups: int = AUTO_GROUPS,
                    trending_n: int = AUTO_TRENDING, ros: dict[str, float] | None = None) -> tuple[list[str], list[str]]:
    pairs = rank_pairs(snap, weekly, trending, bids, pool_size, ros)
    adds = [g[0].add.player_id for g in group_by_add(pairs)[:groups]]
    return adds, trending_free_agents(snap, trending, weekly, trending_n)


# -- add/drop pairs with a positive summed optimal-lineup gain over 'weekly', best first; the candidates are the
#    top 'pool_size' free agents by horizon projection plus every trending one
def rank_pairs(snap: LeagueSnapshot, weekly: list[dict[str, WeekProjection]], trending: dict[str, int] | None = None,
               bids: dict[str, list[float]] | None = None, pool_size: int = DEFAULT_POOL,
               ros: dict[str, float] | None = None) -> list[Pair]:
    pool = candidate_ids(snap, weekly, pool_size)
    extra = [p for p in trending_free_agents(snap, trending or {}, weekly) if p not in pool]
    return _pairs(snap, weekly, pool + extra, trending, bids, positive_only=True, ros=ros)


# -- every drop for one add, best first, whatever the sign of the gain
def pairs_for_add(snap: LeagueSnapshot, weekly: list[dict[str, WeekProjection]], add_id: str,
                  trending: dict[str, int] | None = None, bids: dict[str, list[float]] | None = None,
                  ros: dict[str, float] | None = None) -> list[Pair]:
    return _pairs(snap, weekly, [add_id] if any(add_id in w for w in weekly) else [], trending, bids,
                  positive_only=False, ros=ros)


def _drop_key(p: Pair) -> tuple:
    return -p.gain, p.drop_ros, p.drop_total


def _pairs(snap: LeagueSnapshot, weekly: list[dict[str, WeekProjection]], candidates: list[str],
           trending: dict[str, int] | None, bids: dict[str, list[float]] | None, positive_only: bool,
           ros: dict[str, float] | None = None) -> list[Pair]:
    me = snap.mine
    slots = Slots.from_roster_positions(snap.roster_positions)
    out_of_play = set(me.reserve) | set(me.taxi)
    mine = [p for p in me.players if p not in out_of_play]
    total = lambda pid: horizon_total(weekly, pid)
    # -- without a season-long table the horizon stands in, so ties still break on projection
    rest = (lambda pid: ros.get(pid, 0.0)) if ros is not None else total

    limit = slots.starters + len(slots.ignored) + slots.bench
    drops: list[str | None] = list(mine)
    if len(mine) < limit:
        drops.append(None)

    base = [[w[p] for p in mine if p in w] for w in weekly]
    base_pts = [_lineup_total(pool, slots) for pool in base]
    trending, bids = trending or {}, bids or {}
    pairs = []
    for a in candidates:
        for d in drops:
            gain = 0.0
            for w, pool, b in zip(weekly, base, base_pts):
                if a not in w:
                    continue
                kept = [x for x in pool if x.player_id != d]
                gain += _lineup_total(kept + [w[a]], slots) - b
            if positive_only and gain <= 1e-9:
                continue
            first = next(w for w in weekly if a in w)[a]
            drop_proj = next((w[d] for w in weekly if d in w), None) if d else None
            band = faab_band(bids, first.pos)
            pairs.append(Pair(first, drop_proj, round(gain, 2), round(total(a), 2), int(trending.get(a, 0)),
                              band[0] if band else None, band[1] if band else None,
                              total(d) if d else 0.0, rest(d) if d else 0.0))
    pairs.sort(key=lambda p: (*_drop_key(p), -p.add_total, p.add.player_id, p.drop.player_id if p.drop else ""))
    return pairs


def _band(p: Pair) -> str:
    if p.faab_low is None:
        return ""
    if p.faab_low == p.faab_high:
        return f"${p.faab_low:g}"
    return f"${p.faab_low:g}-${p.faab_high:g}"


def group_by_add(pairs: list[Pair]) -> list[list[Pair]]:
    """Pairs grouped per added player, groups ordered by best gain; the first pair of a group is the recommended one."""
    groups: dict[str, list[Pair]] = {}
    for p in pairs:
        groups.setdefault(p.add.player_id, []).append(p)
    return [sorted(g, key=lambda p: (*_drop_key(p), p.drop.player_id if p.drop else ""))
            for g in sorted(groups.values(), key=lambda g: min(_drop_key(p) for p in g))]


def alternatives(group: list[Pair]) -> list[Pair]:
    return [p for p in group[1:] if group[0].gain - p.gain <= ALT_WINDOW + 1e-9][:MAX_ALTERNATIVES]


def _drop_label(p: Pair) -> str:
    return f"{p.drop.name} ({p.drop.pos}, ros {p.drop_ros:.1f})" if p.drop else "(open roster spot)"


def format_report(week: int, horizon: int, pairs: list[Pair], limit: int = 20, moves: list[str] = ()) -> str:
    lines = [f"Week {week} waivers, lineup gain summed over weeks {week}-{week + horizon - 1}"]
    if moves:
        lines.append("Assumes the reserve moves: " + "; ".join(moves))
    if not pairs:
        lines.append("No add improves the lineup.")
    for g in group_by_add(pairs)[:limit]:
        best = g[0]
        trend = f"  trending +{best.trending_adds:,}" if best.trending_adds else ""
        faab = f"  FAAB {_band(best)}" if _band(best) else ""
        lines.append(f"ADD {best.add.name} ({best.add.pos}) | DROP {_drop_label(best)} | {best.gain:+.1f}{trend}{faab}")
        alts = alternatives(g)
        if alts:
            lines.append("  also: " + ", ".join(f"{_drop_label(p)} {p.gain:+.1f}" for p in alts))
    return "\n".join(lines)


def write_csv(path: str, pairs: list[Pair]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    recommended = {id(g[0]) for g in group_by_add(pairs)}
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "add_id", "add", "pos", "team", "add_proj", "drop_id", "drop", "drop_ros", "gain",
                    "trending_adds", "faab_low", "faab_high", "recommended"])
        for i, p in enumerate(pairs, 1):
            w.writerow([i, p.add.player_id, p.add.name, p.add.pos, p.add.team or "", p.add_total,
                        p.drop.player_id if p.drop else "", p.drop.name if p.drop else "",
                        p.drop_ros if p.drop else "", p.gain,
                        p.trending_adds, "" if p.faab_low is None else p.faab_low,
                        "" if p.faab_high is None else p.faab_high, 1 if id(p) in recommended else 0])


@dataclass
class WeekWaivers:
    horizon: int
    pool_size: int
    players: dict
    weekly: list[dict[str, WeekProjection]]
    trending: dict[str, int]
    bids: dict[str, list[float]]
    pairs: list[Pair]
    ros: dict[str, float]
    snap: LeagueSnapshot
    moves: list[str]


def waivers_for_week(api, snap: LeagueSnapshot, cfg: dict, week: int) -> WeekWaivers:
    """Ranked pairs for the horizon starting at `week`, with the inputs they were ranked from. `snap` in the
    result is my roster after the recommended reserve moves, the roster the pairs were scored on."""
    inseason = cfg.get("inseason") or {}
    horizon = max(1, int(inseason.get("horizon_weeks", DEFAULT_HORIZON)))
    pool_size = int(inseason.get("waiver_pool", DEFAULT_POOL))
    players = free_agent_table(api.players(), snap)
    moved, flags = after_reserve_moves(snap, players)
    rest_weeks = ros_week_count(week)
    season = weekly_projections(api, snap, players, week, max(horizon, rest_weeks), inseason)
    weekly, ros = season[:horizon], rest_of_season(season[:rest_weeks])
    trending, bids = market_inputs(api, str(cfg["league_id"]), week, players)
    return WeekWaivers(horizon, pool_size, players, weekly, trending, bids,
                       rank_pairs(moved, weekly, trending, bids, pool_size, ros), ros, moved,
                       [describe_move(f, players) for f in flags])


def cmd_waivers(cfg, args):
    api, snap = open_league(cfg)
    week = args.week or snap.week
    r = waivers_for_week(api, snap, cfg, week)
    print(format_report(week, r.horizon, r.pairs, moves=r.moves))
    path = os.path.join(cfg["out_dir"], "manage", f"w{week}", "waivers.csv")
    write_csv(path, r.pairs)
    print(f"wrote {path}")
