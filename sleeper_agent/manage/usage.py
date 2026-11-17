"""Usage trends: snap, target and carry shares over the last few completed weeks.

Flags a rostered player whose share fell sharply, and a same-team same-position player who out-snaps
one of mine. Writes one CSV row per player per week.

    python3 -m sleeper_agent usage [--week N] [--candidates ID,ID]
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field

from ..core.league import open_league

DEFAULT_WEEKS = 3
DEFAULT_DROP = 0.15
METRICS = ("snap", "tgt", "car")
FIELDS = {"snap": "off_snp", "tgt": "rec_tgt", "car": "rush_att"}


@dataclass
class Usage:
    player_id: str
    role: str  # -- mine | rival | candidate
    name: str
    pos: str
    team: str
    shares: dict[int, dict[str, float | None]] = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)


def _num(stats: dict, key: str) -> float:
    return float(stats.get(key) or 0)


def week_shares(rows: list) -> dict[str, dict[str, float | None]]:
    """Per player id: snap share and share of the team's targets and carries for one week of stat rows."""
    team_tot: dict[str, dict[str, float]] = {}
    for r in rows:
        team = r.get("team")
        if not team:
            continue
        tot = team_tot.setdefault(team, {"tgt": 0.0, "car": 0.0})
        st = r.get("stats") or {}
        tot["tgt"] += _num(st, FIELDS["tgt"])
        tot["car"] += _num(st, FIELDS["car"])
    out = {}
    for r in rows:
        st = r.get("stats") or {}
        team, pid = r.get("team"), r.get("player_id")
        if not team or pid is None:
            continue
        ratio = lambda num, den: round(num / den, 4) if den > 0 else None
        out[str(pid)] = {
            "snap": ratio(_num(st, FIELDS["snap"]), _num(st, "tm_off_snp")),
            "tgt": ratio(_num(st, FIELDS["tgt"]), team_tot[team]["tgt"]),
            "car": ratio(_num(st, FIELDS["car"]), team_tot[team]["car"]),
        }
    return out


# -- shares for the 'count' most recent weeks before 'week' that have stat rows; empty weeks are skipped
def completed_weeks(fetch, week: int, count: int) -> dict[int, dict]:
    found: dict[int, dict] = {}
    for w in range(week - 1, 0, -1):
        if len(found) >= count:
            break
        rows = fetch(w)
        if rows:
            found[w] = week_shares(rows)
    return dict(sorted(found.items()))


# -- rival id -> the rostered player id it shares a team and position with
def same_team_rivals(players: dict, rostered: list[str]) -> dict[str, str]:
    mine = {pid: players[pid] for pid in rostered if pid in players}
    out: dict[str, str] = {}
    for pid in sorted(players):
        m = players[pid]
        if pid in mine or not m.get("team") or not m.get("pos"):
            continue
        for own in rostered:
            o = mine.get(own)
            if o and o.get("team") == m["team"] and o.get("pos") == m["pos"]:
                out.setdefault(pid, own)
    return out


def usage_table(weekly: dict[int, dict], players: dict, rostered: list[str], candidates: list[str] = (),
                drop: float = DEFAULT_DROP) -> list[Usage]:
    """Rows for the roster, its same-team same-position rivals and the candidates.
    `drop:<metric>`: a share fell by `drop` or more, first week seen to last. `outsnapped:<id>`: a rival's latest snap share beats mine."""
    rivals = same_team_rivals(players, rostered)
    roles = {pid: "candidate" for pid in candidates}
    roles.update({pid: "rival" for pid in rivals})
    roles.update({pid: "mine" for pid in rostered})
    latest = max(weekly) if weekly else None
    rows = []
    for pid, role in roles.items():
        m = players.get(pid) or {}
        u = Usage(pid, role, m.get("name") or pid, m.get("pos") or "", m.get("team") or "")
        u.shares = {w: s[pid] for w, s in weekly.items() if pid in s}
        if role == "mine":
            for metric in METRICS:
                seen = [s[metric] for _, s in sorted(u.shares.items()) if s[metric] is not None]
                if len(seen) >= 2 and seen[0] - seen[-1] >= drop - 1e-9:
                    u.flags.append(f"drop:{metric}")
        rows.append(u)
    by_id = {u.player_id: u for u in rows}
    for pid, own in rivals.items():
        if pid in by_id and by_id[pid].role == "rival" and latest is not None:
            a = by_id[pid].shares.get(latest, {}).get("snap")
            b = by_id[own].shares.get(latest, {}).get("snap") if own in by_id else None
            if a is not None and b is not None and a > b:
                by_id[pid].flags.append(f"outsnapped:{own}")
    order = {"mine": 0, "rival": 1, "candidate": 2}
    rows.sort(key=lambda u: (order[u.role], u.team, u.pos, u.player_id))
    return rows


def format_report(week: int, weeks: list[int], rows: list[Usage]) -> str:
    span = f"weeks {weeks[0]}-{weeks[-1]}" if weeks else "no completed weeks"
    lines = [f"Week {week} usage trends over {span}"]
    flagged = [u for u in rows if u.flags]
    if not flagged:
        lines.append("No usage flags.")
    for u in flagged:
        lines.append(f"{u.role.upper()} {u.name} ({u.pos}, {u.team or 'FA'}): {', '.join(u.flags)}")
    return "\n".join(lines)


def write_csv(path: str, weeks: list[int], rows: list[Usage]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["player_id", "name", "pos", "team", "role", "week", *METRICS, "flags"])
        for u in rows:
            flags = ";".join(u.flags)
            for wk in weeks:
                s = u.shares.get(wk)
                if s is None:
                    continue
                w.writerow([u.player_id, u.name, u.pos, u.team, u.role, wk,
                            *("" if s[m] is None else s[m] for m in METRICS), flags])


def usage_for_week(api, snap, cfg: dict, week: int, candidates: list[str] = ()) -> tuple[list[int], list[Usage]]:
    """-> (the completed weeks the shares cover, the usage rows)."""
    inseason = cfg.get("inseason") or {}
    count = max(1, int(inseason.get("usage_weeks", DEFAULT_WEEKS)))
    drop = float(inseason.get("usage_drop", DEFAULT_DROP))
    weekly = completed_weeks(lambda w: api.week_stats(snap.season, w), week, count)
    return list(weekly), usage_table(weekly, api.players(), list(snap.mine.players), list(candidates), drop)


def cmd_usage(cfg, args):
    api, snap = open_league(cfg)
    week = args.week or snap.week
    candidates = [c for c in (getattr(args, "candidates", None) or "").split(",") if c]
    weeks, rows = usage_for_week(api, snap, cfg, week, candidates)
    print(format_report(week, weeks, rows))
    path = os.path.join(cfg["out_dir"], "manage", f"w{week}", "usage.csv")
    write_csv(path, weeks, rows)
    print(f"wrote {path}")
