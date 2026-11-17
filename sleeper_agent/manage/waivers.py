"""Waiver pairs: add a free agent, drop a rostered player, scored by lineup gain over the next few weeks.

Gain is the change in the best-lineup total summed over the horizon, so a pickup that covers a bye
scores for it. Each add also carries a FAAB band, the 25th to 75th percentile of the league's past
winning bids at that position. Writes a ranked CSV.

    python3 -m sleeper_agent waivers [--week N]
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass

from ..league import LeagueSnapshot, open_league
from ..scoring import Slots
from .lineup import best_lineup, with_defenses
from .projections import WeekProjection, project_week

DEFAULT_POOL = 40
DEFAULT_HORIZON = 3
ALT_WINDOW = 0.1
MAX_ALTERNATIVES = 3


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


def rank_pairs(snap: LeagueSnapshot, weekly: list[dict[str, WeekProjection]], trending: dict[str, int] | None = None,
               bids: dict[str, list[float]] | None = None, pool_size: int = DEFAULT_POOL) -> list[Pair]:
    """Add/drop pairs with a positive summed optimal-lineup gain over `weekly`, best first.
    A free agent is a player on no roster; the candidates are the top `pool_size` by horizon projection."""
    me = snap.mine
    slots = Slots.from_roster_positions(snap.roster_positions)
    rostered = {p for r in snap.rosters for p in (*r.players, *r.reserve, *r.taxi)}
    out_of_play = set(me.reserve) | set(me.taxi)
    mine = [p for p in me.players if p not in out_of_play]
    ids = set().union(*(w.keys() for w in weekly)) if weekly else set()
    total = lambda pid: sum(w[pid].pts for w in weekly if pid in w)
    candidates = sorted((p for p in ids if p not in rostered and total(p) > 0), key=lambda p: (-total(p), p))[:pool_size]

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
            if gain <= 1e-9:
                continue
            first = next(w for w in weekly if a in w)[a]
            drop_proj = next((w[d] for w in weekly if d in w), None) if d else None
            band = faab_band(bids, first.pos)
            pairs.append(Pair(first, drop_proj, round(gain, 2), round(total(a), 2), int(trending.get(a, 0)),
                              band[0] if band else None, band[1] if band else None,
                              total(d) if d else 0.0))
    pairs.sort(key=lambda p: (-p.gain, p.drop_total, -p.add_total,
                              p.add.player_id, p.drop.player_id if p.drop else ""))
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
    return [sorted(g, key=lambda p: (-p.gain, p.drop_total, p.drop.player_id if p.drop else ""))
            for g in sorted(groups.values(), key=lambda g: min((-p.gain, p.drop_total) for p in g))]


def _drop_label(p: Pair) -> str:
    return f"{p.drop.name} ({p.drop.pos})" if p.drop else "(open roster spot)"


def format_report(week: int, horizon: int, pairs: list[Pair], limit: int = 20) -> str:
    lines = [f"Week {week} waivers, lineup gain summed over weeks {week}-{week + horizon - 1}"]
    if not pairs:
        lines.append("No add improves the lineup.")
    for g in group_by_add(pairs)[:limit]:
        best = g[0]
        trend = f"  trending +{best.trending_adds:,}" if best.trending_adds else ""
        faab = f"  FAAB {_band(best)}" if _band(best) else ""
        lines.append(f"ADD {best.add.name} ({best.add.pos}) | DROP {_drop_label(best)} | {best.gain:+.1f}{trend}{faab}")
        alts = [p for p in g[1:] if best.gain - p.gain <= ALT_WINDOW + 1e-9][:MAX_ALTERNATIVES]
        if alts:
            lines.append("  also: " + ", ".join(f"{_drop_label(p)} {p.gain:+.1f}" for p in alts))
    return "\n".join(lines)


def write_csv(path: str, pairs: list[Pair]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    recommended = {id(g[0]) for g in group_by_add(pairs)}
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "add_id", "add", "pos", "team", "add_proj", "drop_id", "drop", "gain",
                    "trending_adds", "faab_low", "faab_high", "recommended"])
        for i, p in enumerate(pairs, 1):
            w.writerow([i, p.add.player_id, p.add.name, p.add.pos, p.add.team or "", p.add_total,
                        p.drop.player_id if p.drop else "", p.drop.name if p.drop else "", p.gain,
                        p.trending_adds, "" if p.faab_low is None else p.faab_low,
                        "" if p.faab_high is None else p.faab_high, 1 if id(p) in recommended else 0])


def cmd_waivers(cfg, args):
    api, snap = open_league(cfg)
    lid = str(cfg["league_id"])
    inseason = cfg.get("inseason") or {}
    week = args.week or snap.week
    horizon = max(1, int(inseason.get("horizon_weeks", DEFAULT_HORIZON)))
    pool_size = int(inseason.get("waiver_pool", DEFAULT_POOL))
    byes = api.byes(snap.season)
    players = free_agent_table(api.players(), snap)
    extra = {"DEF"} & set(snap.roster_positions)
    weekly = []
    for w in range(week, week + horizon):
        bye_teams = {t for t, bw in byes.items() if bw == w}
        weekly.append(project_week(api.week_projections(snap.season, w, extra), players, snap.scoring_settings,
                                   bye_teams, inseason))
    trending = {str(t.get("player_id")): int(t.get("count") or 0) for t in api.trending("add")}
    past = [t for r in range(1, week + 1) for t in api.transactions(lid, r)]
    pairs = rank_pairs(snap, weekly, trending, winning_bids(past, players), pool_size)
    print(format_report(week, horizon, pairs))
    path = os.path.join(cfg["out_dir"], "manage", f"w{week}", "waivers.csv")
    write_csv(path, pairs)
    print(f"wrote {path}")
