"""The weekly manage packet: the lineup, injury, waiver and usage results for one week, joined with the latest
in-season research set, as one JSON-serializable dict. `closed_set` names every player the packet mentions,
so anything written from it can be checked against that list.

    python3 -m ball_knower manage [--week N] [--watch NAME]
"""
from __future__ import annotations

import sys

from ..core.data import norm_name
from .injuries import flags_for_week
from .lineup import lineup_for_week
from .projections import WeekProjection
from .refresh import DEFAULT_MAX_AGE_DAYS, empty_current, load_current
from .usage import METRICS, usage_for_week
from .waivers import (ALT_WINDOW, Pair, alternatives, auto_candidates, free_agent_table, group_by_add, horizon_total,
                      pairs_for_add, rostered_ids, waivers_for_week)

GLOSSARY = {
    "slot": "current roster slot: a starting position, BN, IR or TAXI",
    "proj": "this week's projected points under league scoring, after bye and injury adjustments",
    "ros": "projected points summed from the report week through week 17 under the same adjustments; a "
           "Questionable, Doubtful or Out tag adjusts only the current NFL week",
    "note": "the adjustment applied to proj: bye | out | questionable | doubtful",
    "status": "Sleeper injury tag",
    "bye": "the team's bye week",
    "usage": "latest completed week's snap share and the player's share of team targets (tgt) and carries (car)",
    "usage_flags": "drop:<metric>: that share fell by the configured drop or more across the usage weeks",
    "outsnapped_by": "same-team same-position players whose latest snap share beats this one",
    "role": "research role: LOCKED | SHARED | CONTESTED | UNKNOWN",
    "risk": "research risk, 0 (safe) to 5 (expected to miss games)",
    "tags": "research tags",
    "gain": "change in best-lineup points summed over the horizon weeks if the move is made",
    "horizon_proj": "projected points summed over the horizon weeks",
    "trending_adds": "Sleeper-wide adds over the last 24 hours",
    "faab_low, faab_high": "25th and 75th percentile of the league's past winning bids at the position",
    "drop": "recommended drop, the lowest ros among drops tied on gain; null is an open roster spot",
    "drop_order": "players outside the report week's best lineup and off reserve and taxi, lowest ros first",
    "notes": "free text from the manager, named with --note",
    "adds_assume_moves": "reserve moves from injuries the adds count as made: ir_open on reserve, ir_cleared back in play",
    "alternatives": f"other drops within {ALT_WINDOW:g} points of the best gain",
    "watched": "named with --watch; its gain may be zero or negative",
    "best_gain": "best gain over every drop; 0 when no drop makes the add positive",
    "replay": "present only on a replay of a past week: rosters and FAAB are rebuilt to `cutoff`; injury tags, "
              "lineups, reserve lists and trending counts are unknown, every player counts as healthy, no research "
              "set is read, and passing_on is empty",
}


def resolve_watch(table: dict, names) -> list[str]:
    """Player ids for `names` matched through `norm_name`; exits on a name with no match.
    Among namesakes, a player with a team wins, then the lower search rank."""
    index: dict[str, list[str]] = {}
    for pid, m in table.items():
        index.setdefault(norm_name(m.get("name") or pid), []).append(pid)
    out = []
    for name in names:
        hits = index.get(norm_name(name))
        if not hits:
            sys.exit(f"--watch {name!r}: no player of that name")
        out.append(min(hits, key=lambda p: (not table[p].get("team"), table[p].get("search_rank") or 10**9, p)))
    return list(dict.fromkeys(out))


def _who(table: dict, pid: str) -> dict:
    m = table.get(pid) or {}
    return {"name": m.get("name") or pid, "pos": m.get("pos") or "", "team": m.get("team")}


def _proj_row(p: WeekProjection) -> dict:
    return {"name": p.name, "pos": p.pos, "team": p.team, "pts": p.pts, "note": p.note}


def _drop_row(p: Pair) -> dict | None:
    if p.drop is None:
        return None
    return {"name": p.drop.name, "pos": p.drop.pos, "team": p.drop.team, "horizon_proj": round(p.drop_total, 2),
            "ros": round(p.drop_ros, 2)}


def _add_row(group: list[Pair], watched: bool, week: dict[str, WeekProjection], byes: dict[str, int]) -> dict:
    best = group[0]
    now = week.get(best.add.player_id)
    return {"name": best.add.name, "pos": best.add.pos, "team": best.add.team, "status": best.add.status,
            "proj": now.pts if now else None, "bye": byes.get(best.add.team), "gain": best.gain,
            "horizon_proj": best.add_total, "trending_adds": best.trending_adds,
            "faab_low": best.faab_low, "faab_high": best.faab_high, "drop": _drop_row(best),
            "alternatives": [{"drop": _drop_row(p), "gain": p.gain} for p in alternatives(group)],
            "watched": watched}


def _slots(snap) -> dict[str, str]:
    me = snap.mine
    # -- Sleeper lists starters in the order of the starting slots of roster_positions; "0" is an empty slot
    labels = [p for p in snap.roster_positions if p not in ("BN", "IR", "TAXI")]
    out = {pid: lab for pid, lab in zip(me.starters, labels) if pid and pid != "0"}
    out.update({pid: "IR" for pid in me.reserve})
    out.update({pid: "TAXI" for pid in me.taxi})
    return out


def _research_lookup(df):
    index: dict[str, list[dict]] = {}
    for r in df.to_dict("records"):
        index.setdefault(norm_name(r["name"]), []).append(r)

    def find(name: str, team: str | None) -> dict:
        rows = index.get(norm_name(name)) or [{}]
        return next((r for r in rows if r.get("team") == team), rows[0])
    return find


def _roster_rows(snap, table, lineup, weeks, usage_rows, byes, research, ros) -> list[dict]:
    slots = _slots(snap)
    usage = {u.player_id: u for u in usage_rows if u.role == "mine"}
    latest = weeks[-1] if weeks else None
    beaten: dict[str, list[str]] = {}
    for u in usage_rows:
        for f in u.flags:
            if f.startswith("outsnapped:"):
                beaten.setdefault(f.split(":", 1)[1], []).append(u.name)
    rows = []
    for pid in snap.mine.players:
        who = _who(table, pid)
        proj = lineup.proj.get(pid)
        u = usage.get(pid)
        shares = u.shares.get(latest) if u and latest is not None else None
        res = research(who["name"], who["team"])
        rows.append({**who, "slot": slots.get(pid, "BN"),
                     "proj": proj.pts if proj else None, "note": proj.note if proj else "",
                     "ros": ros.get(pid),
                     "status": (table.get(pid) or {}).get("injury_status"), "bye": byes.get(who["team"]),
                     "usage": {"week": latest, **{m: shares[m] for m in METRICS}} if shares else None,
                     "usage_flags": list(u.flags) if u else [], "outsnapped_by": beaten.get(pid, []),
                     "role": res.get("role") or None, "risk": res.get("risk") or None,
                     "tags": [t for t in str(res.get("tags") or "").split("|") if t]})
    return rows


def build_packet(api, snap, cfg: dict, week: int, notes=(), watch=(), replay: dict | None = None) -> dict:
    """Every fact the weekly report draws on. Adds are the best add groups plus each watched free agent,
    scored on the roster after the recommended reserve moves. `passing_on` is the trending free agents not
    among the adds. A `replay` reads no research set and leaves `passing_on` empty."""
    watch_ids = resolve_watch(free_agent_table(api.players(), snap), watch)
    lineup = lineup_for_week(api, snap, cfg, week)
    flags = flags_for_week(api, snap, cfg, week)
    wv = waivers_for_week(api, snap, cfg, week)
    weeks, usage_rows = usage_for_week(api, snap, cfg, week)
    max_age = float((cfg.get("research") or {}).get("max_age_days", DEFAULT_MAX_AGE_DAYS))
    current = empty_current() if replay else load_current(cfg["data_dir"], max_age)
    table, me = wv.players, snap.mine
    byes = api.byes(snap.season)

    moved = wv.snap
    add_ids, trend_ids = auto_candidates(moved, wv.weekly, wv.trending, wv.bids, wv.pool_size, ros=wv.ros)
    groups = {g[0].add.player_id: g for g in group_by_add(wv.pairs)}
    adds = [_add_row(groups[a], a in watch_ids, wv.weekly[0], byes) for a in add_ids]
    rostered = rostered_ids(snap)
    for pid in (p for p in watch_ids if p not in rostered and p not in add_ids):
        forced = group_by_add(pairs_for_add(moved, wv.weekly, pid, wv.trending, wv.bids, wv.ros))
        if forced:
            adds.append(_add_row(forced[0], True, wv.weekly[0], byes))
            add_ids.append(pid)

    if replay:
        trend_ids = []
    passing_on = []
    for pid in (t for t in trend_ids if t not in add_ids):
        best = max((p.gain for p in pairs_for_add(moved, wv.weekly, pid, wv.trending, wv.bids, wv.ros)), default=0.0)
        passing_on.append({**_who(table, pid), "trending_adds": wv.trending.get(pid, 0), "best_gain": max(best, 0.0),
                           "horizon_proj": round(horizon_total(wv.weekly, pid), 2)})

    # -- the starters set in Sleeper may be last week's, so the report week's best lineup decides who is safe
    placed = {p.player_id for p in lineup.best} | set(me.reserve) | set(me.taxi)
    bench = sorted((p for p in me.players if p not in placed), key=lambda p: (wv.ros.get(p, 0.0), p))
    drop_order = [{**_who(table, p), "horizon_proj": round(horizon_total(wv.weekly, p), 2), "ros": wv.ros.get(p, 0.0)}
                  for p in bench]

    owners = {pid: ("me" if r.is_mine else r.owner) for r in snap.rosters for pid in (*r.players, *r.reserve, *r.taxi)}
    watched = [{**_who(table, p), "owner": owners.get(p, "free agent")} for p in watch_ids]

    roster = _roster_rows(snap, table, lineup, weeks, usage_rows, byes, _research_lookup(current.players), wv.ros)
    injuries = [{"kind": f.kind, **_who(table, f.player_id), "status": (table.get(f.player_id) or {}).get("injury_status"),
                 "text": f.text} for f in flags]
    best_rows, current_rows = [_proj_row(p) for p in lineup.best], [_proj_row(p) for p in lineup.current]
    swaps = [{"start": _proj_row(s.start), "sit": _proj_row(s.sit) if s.sit else None, "delta": round(s.delta, 2)}
             for s in lineup.swaps]

    names = {r["name"] for r in (*roster, *best_rows, *current_rows, *injuries, *adds, *passing_on, *drop_order,
                                 *watched)}
    names |= {n for r in roster for n in r["outsnapped_by"]}
    names |= {d["name"] for a in adds for d in (a["drop"], *(x["drop"] for x in a["alternatives"])) if d}
    keys = {norm_name(n) for n in names}
    news = [] if replay else [r for r in current.news.to_dict("records") if norm_name(r["name"]) in keys]

    packet = {
        "league": f"{snap.name}: {snap.season} {snap.season_type}",
        "week": week,
        "horizon_weeks": [week, week + wv.horizon - 1],
        "record": None if replay else f"{me.wins}-{me.losses}-{me.ties}",
        "faab": {"left": me.faab_left, "budget": snap.faab_budget},
        "roster": roster,
        "lineup": {"best": best_rows, "current": current_rows, "swaps": swaps,
                   "best_total": round(sum(p.pts for p in lineup.best), 2),
                   "current_total": round(sum(p.pts for p in lineup.current), 2)},
        "injuries": injuries,
        "adds": adds,
        "adds_assume_moves": wv.moves,
        "passing_on": passing_on,
        "drop_order": drop_order,
        "news": news,
        "research": {"set": current.name, "date": current.date, "warning": current.warning},
        "notes": list(notes),
        "watch": watched,
        "closed_set": sorted(names),
        "glossary": GLOSSARY,
    }
    if replay:
        packet["replay"] = dict(replay)
    return packet
