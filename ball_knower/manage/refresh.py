"""In-season research refresh: a one-week news sweep per NFL team that matters to the next roster moves.

A shard covers one team. Its slice lists my players on the team (`mine`), the waiver candidates on it
(`candidate`) and their same-team same-position rivals (`rival`). Replies merge into
data/research/<season>-w<N>/players.csv and news.csv, and current.json names the latest set; a previous
row whose player the new replies hold no row for in that table is carried forward with its original as_of.
`auto` takes the adds of the best waiver groups plus the most-added trending free agents as candidates;
ALL and a team list take the whole pool plus every trending free agent, a list keeping only its teams.

    python3 -m ball_knower research --in-season [--week N] [--teams auto|ALL|ARI,BUF]
                                      [--write-only | --from-replies] [--parallel N] [--model M] [--timeout S]
"""
from __future__ import annotations

import datetime
import io
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import pandas as pd

from ..core.data import norm_name
from ..core.league import open_league
from ..research.shards import _slug, run_shard
from .usage import same_team_rivals
from .waivers import (DEFAULT_HORIZON, DEFAULT_POOL, auto_candidates, candidate_ids, free_agent_table, market_inputs,
                      trending_free_agents, weekly_projections)

PLAYER_COLS = ["name", "pos", "team", "role", "risk", "inj", "tags", "as_of"]
NEWS_COLS = ["name", "team", "status", "note", "as_of", "source"]
SLICE_COLS = ["name", "pos", "team", "mark", "status"]
MARKS = ("mine", "candidate", "rival")
DEFAULT_MAX_AGE_DAYS = 7

RULES = """Rules for the columns:
- role: LOCKED (unambiguous starter, >=65% of position's opportunity) | SHARED (committee/1B) | CONTESTED (job in question) | UNKNOWN.
- risk 0-5 int: 0 locked+healthy+prime age; 1 one minor concern; 2 moderate injury history OR mild role competition; 3 active injury designation OR real role competition OR age cliff (RB>=28, WR>=31, TE>=32); 4 two of those; 5 holdout/suspension/PUP/expected to miss >=3 games.
- inj: OK | Q | D | O | PUP | HOLDOUT | SUSP.
- tags: 0-3 per row, |-separated, from exactly: NEW_OC, NEW_QB, NEW_TEAM, ROOKIE, ROOKIE_THREAT, HANDCUFF, RZ_HOG, YAC, DEEP, VOLUME, CONTRACT_YR, AGE_CLIFF, IR_STASH, INJ_HISTORY, BREAKOUT, DECLINE.
- news.csv: name and team exactly as in the slice; status is the availability the item implies (OK | Q | D | O | IR | PUP | SUSP); note is one sentence; as_of is the item's date, YYYY-MM-DD; source is the outlet or URL."""

PROMPT = """You are one shard of a mid-season fantasy football research job. You own one NFL team and no other:

  {team}

It is week {week} of the {season} season and today is {today}. League: {league}

Below is your slice: my players on {team} (mark mine), free agents I may add (mark candidate) and the players who share a team and position with one of them (mark rival). Cover only these rows: do not add players, do not touch other teams.

Your job:
1. If you have web access: news sweep, last 7 days, for {team} - injuries and practice status, depth chart and role changes, snap, target and carry trends, trades, suspensions, coaching changes. No web access: use what you know; leave a field blank rather than guess.
2. Emit one players.csv row for every slice row, the role, risk, inj and tags columns reflecting the sweep.
3. Emit one news.csv row per concrete item; a player with no news gets none.

{rules}

OUTPUT: exactly two fenced code blocks, each opened with the filename as the info string, nothing else in your reply:
```players.csv
name,pos,team,role,risk,inj,tags
<every row from your slice>
```
```news.csv
name,team,status,note,as_of,source
<one row per news item>
```

YOUR SLICE:
```csv
{slice_csv}
```"""

BLOCK_RE = re.compile(r"```(players\.csv|news\.csv)\s*\n(.*?)```", re.S)
SLICE_RE = re.compile(r"YOUR SLICE:\s*```csv\s*\n(.*?)```", re.S)
SET_RE = re.compile(r"(\d{4})-w(\d+)")
PROMPT_RE = re.compile(r"shard_[A-Za-z0-9_]+_prompt\.md")


@dataclass
class Current:
    players: pd.DataFrame
    news: pd.DataFrame
    name: str | None
    date: str | None
    teams: list[str]
    warning: str | None


def empty_current(warning: str | None = None) -> Current:
    return Current(pd.DataFrame(columns=PLAYER_COLS), pd.DataFrame(columns=NEWS_COLS), None, None, [], warning)


def build_slices(table: dict, mine: list[str], candidates: list[str]) -> dict[str, pd.DataFrame]:
    """Per NFL team, the rows of its shard. A player that is mine and also a rival of a candidate is `mine`."""
    marks = {pid: "rival" for pid in same_team_rivals(table, candidates)}
    marks.update({pid: "rival" for pid in same_team_rivals(table, mine)})
    marks.update({pid: "candidate" for pid in candidates})
    marks.update({pid: "mine" for pid in mine})
    rows = []
    for pid, mark in marks.items():
        m = table.get(pid) or {}
        if m.get("team") and m.get("pos"):
            rows.append({"name": m.get("name") or pid, "pos": m["pos"], "team": m["team"], "mark": mark,
                         "status": m.get("injury_status") or ""})
    rows.sort(key=lambda r: (MARKS.index(r["mark"]), r["pos"], r["name"]))
    df = pd.DataFrame(rows, columns=SLICE_COLS)
    return {team: g.reset_index(drop=True) for team, g in df.groupby("team")}


def pick_teams(spec: str | None, slices: dict) -> tuple[list[str], list[str]]:
    """-> (teams to shard, notes). `auto` and `ALL` take every team with rows; a list takes its teams that have rows."""
    spec = spec.strip()
    if spec.lower() in ("auto", "all"):
        return sorted(slices), []
    wanted = [t.strip().upper() for t in spec.split(",") if t.strip()]
    notes = [f"no rows for team {t}: skipped" for t in wanted if t not in slices]
    return sorted({t for t in wanted if t in slices}), notes


def write_prompts(set_dir: str, slices: dict, teams: list[str], season, week: int, league: str, today: str) -> list[str]:
    os.makedirs(set_dir, exist_ok=True)
    for f in os.listdir(set_dir):
        if PROMPT_RE.fullmatch(f):
            os.remove(os.path.join(set_dir, f))
    paths = []
    for i, team in enumerate(teams, 1):
        prompt = PROMPT.format(team=team, week=week, season=season, today=today, league=league, rules=RULES,
                               slice_csv=slices[team].to_csv(index=False).strip())
        path = os.path.join(set_dir, f"shard_{i:02d}_{_slug(team)}_prompt.md")
        with open(path, "w") as f:
            f.write(prompt)
        paths.append(path)
    return paths


def parse_reply(text: str) -> dict[str, pd.DataFrame]:
    out = {}
    for name, body in BLOCK_RE.findall(text):
        try:
            out[name] = pd.read_csv(io.StringIO(body.strip()), dtype=str, keep_default_na=False)
        except Exception:  # noqa
            pass
    return out


def find_set(root: str, week: int | None = None) -> str | None:
    if not os.path.isdir(root):
        return None
    found = []
    for d in os.listdir(root):
        m = SET_RE.fullmatch(d)
        if m and os.path.isdir(os.path.join(root, d)) and (week is None or int(m[2]) == int(week)):
            found.append((int(m[1]), int(m[2]), d))
    return max(found)[2] if found else None


def collect(set_dir: str) -> tuple[dict[str, pd.DataFrame], dict[str, dict]]:
    slices, replies = {}, {}
    for f in sorted(os.listdir(set_dir)):
        if not PROMPT_RE.fullmatch(f):
            continue
        with open(os.path.join(set_dir, f)) as fh:
            m = SLICE_RE.search(fh.read())
        if not m:
            continue
        sl = pd.read_csv(io.StringIO(m[1].strip()), dtype=str, keep_default_na=False)
        if sl.empty:
            continue
        team = sl["team"].iloc[0]
        slices[team] = sl
        for ext in ("md", "txt"):
            reply = os.path.join(set_dir, f.replace("_prompt.md", f"_reply.{ext}"))
            if os.path.exists(reply):
                with open(reply) as fh:
                    replies[team] = parse_reply(fh.read())
                break
    return slices, replies


def _read_table(path: str, cols: list[str]) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=cols)
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    return df[cols]


def _read_info(data_dir: str) -> dict | None:
    try:
        with open(os.path.join(data_dir, "research", "current.json")) as f:
            info = json.load(f)
    except (OSError, ValueError):
        return None
    return info if isinstance(info, dict) and info.get("set") else None


def merge(data_dir: str, set_name: str, slices: dict, replies: dict, today: str) -> list[str]:
    """Write the set and current.json from shard replies; returns notes. A reply row naming a player outside
    its shard's slice is dropped; previous-set rows stay for players the replies hold no row for in that table."""
    notes, players, news, refreshed = [], {}, [], set()
    for team, rep in sorted(replies.items()):
        canon = {norm_name(n): (n, p) for n, p in zip(slices[team]["name"], slices[team]["pos"])}
        dropped = []
        for table in ("players.csv", "news.csv"):
            df = rep.get(table)
            if df is None or "name" not in df.columns:
                notes.append(f"shard {team}: no parsable {table} block")
                continue
            refreshed.add(team)
            for r in df.to_dict("records"):
                hit = canon.get(norm_name(r["name"]))
                if hit is None:
                    dropped.append(str(r["name"]))
                    continue
                name, pos = hit
                cell = lambda c: str(r.get(c, "")).strip()
                if table == "players.csv":
                    players[norm_name(name)] = {"name": name, "pos": pos, "team": team, "role": cell("role"),
                                                "risk": cell("risk"), "inj": cell("inj"), "tags": cell("tags"),
                                                "as_of": today}
                elif cell("note") or cell("status"):
                    news.append({"name": name, "team": team, "status": cell("status"), "note": cell("note"),
                                 "as_of": cell("as_of") or today, "source": cell("source")})
        if dropped:
            notes.append(f"shard {team}: dropped {len(dropped)} row(s) naming players outside its slice: {sorted(set(dropped))}")
    if not refreshed:
        sys.exit("no shard reply held a parsable players.csv or news.csv block")

    root = os.path.join(data_dir, "research")
    prev = _read_info(data_dir)
    prev_players, prev_news = (pd.DataFrame(columns=PLAYER_COLS), pd.DataFrame(columns=NEWS_COLS))
    if prev:
        prev_players = _read_table(os.path.join(root, prev["set"], "players.csv"), PLAYER_COLS)
        prev_news = _read_table(os.path.join(root, prev["set"], "news.csv"), NEWS_COLS)
        for df in (prev_players, prev_news):
            df.loc[df["as_of"] == "", "as_of"] = str(prev.get("date") or "")
    keep_p = prev_players[~prev_players["name"].map(norm_name).isin(set(players))]
    keep_n = prev_news[~prev_news["name"].map(norm_name).isin({norm_name(r["name"]) for r in news})]
    if prev and (len(keep_p) or len(keep_n)):
        notes.append(f"carried forward {len(keep_p)} player row(s) and {len(keep_n)} news row(s) from {prev['set']}")

    out_p = pd.concat([pd.DataFrame(list(players.values()), columns=PLAYER_COLS), keep_p], ignore_index=True)
    out_n = pd.concat([pd.DataFrame(news, columns=NEWS_COLS), keep_n], ignore_index=True).drop_duplicates()
    set_dir = os.path.join(root, set_name)
    os.makedirs(set_dir, exist_ok=True)
    out_p.sort_values(["team", "name"]).to_csv(os.path.join(set_dir, "players.csv"), index=False)
    out_n.sort_values(["team", "name", "as_of"]).to_csv(os.path.join(set_dir, "news.csv"), index=False)
    with open(os.path.join(root, "current.json"), "w") as f:
        json.dump({"set": set_name, "date": today, "teams": sorted(refreshed)}, f, indent=1)
    return notes


def load_current(data_dir: str, max_age_days: float = DEFAULT_MAX_AGE_DAYS, today: str | None = None) -> Current:
    """The latest refresh set. `warning` is set when no set exists or the set is older than `max_age_days`."""
    info = _read_info(data_dir)
    if info is None:
        return empty_current("no in-season research set; run `research --in-season`")
    set_dir = os.path.join(data_dir, "research", info["set"])
    cur = Current(_read_table(os.path.join(set_dir, "players.csv"), PLAYER_COLS),
                  _read_table(os.path.join(set_dir, "news.csv"), NEWS_COLS),
                  info["set"], info.get("date"), list(info.get("teams") or []), None)
    try:
        now = datetime.date.fromisoformat(today) if today else datetime.date.today()
        age = (now - datetime.date.fromisoformat(str(info["date"]))).days
    except (KeyError, ValueError):
        cur.warning = f"research set {cur.name} has no readable date"
        return cur
    if age > float(max_age_days):
        cur.warning = f"research set {cur.name} is {age} days old (limit {max_age_days:g}); run `research --in-season`"
    return cur


def _prepare(cfg: dict, args, root: str, today: str) -> tuple[str, list[str]]:
    api, snap = open_league(cfg)
    inseason = cfg.get("inseason") or {}
    week = int(args.week or snap.week)
    horizon = max(1, int(inseason.get("horizon_weeks", DEFAULT_HORIZON)))
    pool = int(inseason.get("waiver_pool", DEFAULT_POOL))
    table = free_agent_table(api.players(), snap)
    weekly = weekly_projections(api, snap, table, week, horizon, inseason)
    trending, bids = market_inputs(api, str(cfg["league_id"]), week, table)
    spec = args.teams
    if spec.strip().lower() == "auto":
        adds, trend = auto_candidates(snap, weekly, trending, bids, pool)
    else:
        adds, trend = candidate_ids(snap, weekly, pool), trending_free_agents(snap, trending, weekly)
    slices = build_slices(table, list(snap.mine.players), list(dict.fromkeys(adds + trend)))
    teams, notes = pick_teams(spec, slices)
    for note in notes:
        print("NOTE:", note)
    if not teams:
        sys.exit("no team has rows to research")
    set_name = f"{snap.season}-w{week}"
    paths = write_prompts(os.path.join(root, set_name), slices, teams, snap.season, week, snap.name, today)
    return set_name, paths


def _run(cfg: dict, args, prompts: list[str]) -> None:
    llm = cfg.get("llm", {})
    model = args.model or llm.get("research_model") or ""
    timeout = float(args.timeout) if args.timeout is not None else float(llm.get("research_timeout_s") or 900)
    backend = llm.get("backend", "claude-cli")
    if backend == "none":
        sys.exit("llm.backend is 'none' - research needs an LLM; set llm.backend or use --write-only")
    print(f"running {len(prompts)} shards via {backend} / {model or 'backend default'} (parallel {args.parallel}) ...")

    def one(p):
        reply = p.replace("_prompt.md", "_reply.md")
        for stale in (reply, reply.replace("_reply.md", "_reply.txt")):
            if os.path.exists(stale):
                os.remove(stale)
        try:
            text = run_shard(p, dict(llm, backend=backend), model, timeout)
            with open(reply, "w") as f:
                f.write(text)
            print(f"  done {os.path.basename(p)}")
        except Exception as e:  # noqa
            print(f"  FAIL {os.path.basename(p)}: {e}")

    with ThreadPoolExecutor(max_workers=int(args.parallel)) as ex:
        list(ex.map(one, prompts))


def cmd_refresh(cfg, args):
    root = os.path.join(cfg["out_dir"], "research")
    today = datetime.date.today().isoformat()
    if args.from_replies:
        set_name = find_set(root, args.week)
        if not set_name:
            sys.exit(f"no in-season prompt set in {root}; run `research --in-season --write-only` first")
    else:
        set_name, prompts = _prepare(cfg, args, root, today)
        print(f"wrote {len(prompts)} shard prompts to {os.path.join(root, set_name)}/")
        if args.write_only:
            print("run each in any LM; save each reply beside its prompt with _prompt -> _reply in the "
                  "name (e.g. shard_01_ARI_reply.md); then: research --in-season --from-replies")
            return 0
        _run(cfg, args, prompts)
    slices, replies = collect(os.path.join(root, set_name))
    if not replies:
        sys.exit(f"no shard replies matching the prompts in {os.path.join(root, set_name)}")
    for note in merge(cfg["data_dir"], set_name, slices, replies, today):
        print("NOTE:", note)
    print(f"merged -> {os.path.join(cfg['data_dir'], 'research', set_name)}/players.csv, news.csv")
    return 0
