"""Build data/players_skeleton.csv from Sleeper's projections + schedule.

One API call fills every numeric column: proj (raw stat projections scored with
the league's exact scoring settings), Sleeper ADP, team, bye, age, and names
that resolve to Sleeper ids exactly. The LLM research pass then only fills the
judgment columns (floor/ceil/risk/role/inj/po_sched/tags) and the stacks/anti
tables — no bulk data collection.
"""
from __future__ import annotations

import datetime
import json
import os

import numpy as np
import pandas as pd

from .data import PLAYER_COLS, POS
from .sleeper import Sleeper

COUNTS = {"QB": 24, "RB": 70, "WR": 80, "TE": 24}
DEFAULT_SCORING = {"pass_td": 4, "pass_yd": 0.04, "pass_int": -1, "pass_2pt": 2,
                   "rush_td": 6, "rush_yd": 0.1, "rush_2pt": 2,
                   "rec": 0.5, "rec_td": 6, "rec_yd": 0.1, "rec_2pt": 2, "fum_lost": -2}
INJ_MAP = {"Questionable": "Q", "Doubtful": "D", "Out": "O", "PUP": "PUP", "IR": "O", "Sus": "SUSP", "COV": "O", "NA": "O", "DNR": "O"}


def gap_tiers(proj: np.ndarray, top_k: int = 40, min_gap: float = 4.0) -> list[int]:
    """Tier breaks at gaps larger than the 80th percentile of gaps in the top of the board."""
    if len(proj) < 3:
        return [1] * len(proj)
    gaps = -np.diff(proj)
    thr = max(min_gap, float(np.quantile(gaps[: min(top_k, len(gaps))], 0.8)))
    tiers, t = [1], 1
    for g in gaps:
        if g > thr:
            t += 1
        tiers.append(t)
    return tiers


def build_skeleton(data_dir: str, season: int | None = None, scoring: dict | None = None,
                   counts: dict | None = None) -> tuple[pd.DataFrame, list[str]]:
    notes = []
    api = Sleeper(data_dir)
    lj = os.path.join(data_dir, "league.json")
    if (scoring is None or season is None) and os.path.exists(lj):
        with open(lj) as f:
            info = json.load(f)
        scoring = scoring or info.get("scoring_settings")
        season = season or int(info.get("season") or 0) or None
    if season is None:
        season = datetime.date.today().year
        notes.append(f"season not in league.json; assuming {season}")
    if not scoring:
        scoring = DEFAULT_SCORING
        notes.append("no league.json scoring; using default half-PPR — run `draftkit setup` first for exact points")

    rec = float(scoring.get("rec", 0))
    adp_field = "adp_ppr" if rec >= 1 else ("adp_half_ppr" if rec > 0 else "adp_std")
    players = api.players()
    proj = api.projections(season)
    try:
        byes = api.byes(season)
    except Exception as e:  # noqa
        byes = {}
        notes.append(f"schedule endpoint failed ({type(e).__name__}); bye column left at 0 — judge pass must fill it")

    rows = []
    for p in proj:
        st = p.get("stats") or {}
        pid = str(p.get("player_id"))
        meta = players.get(pid) or {}
        pos = meta.get("pos") or (p.get("player") or {}).get("position")
        team = p.get("team") or meta.get("team")
        if pos not in POS or not team:
            continue
        pts = sum(float(v) * float(scoring[k]) for k, v in st.items() if k in scoring and isinstance(v, (int, float)))
        if pts <= 0:
            continue
        adp = float(st.get(adp_field) or 999.0)
        pl = p.get("player") or {}
        name = meta.get("name") or f"{pl.get('first_name','')} {pl.get('last_name','')}".strip()
        inj = INJ_MAP.get(meta.get("injury_status") or "", "OK")
        rows.append({"name": name, "pos": pos, "team": team, "bye": int(byes.get(team, 0)),
                     "proj": round(pts, 1), "floor": 0.0, "ceil": 0.0, "tier": 0, "adp": adp,
                     "risk": 2, "role": "UNKNOWN", "inj": inj, "age": int(meta.get("age") or 0),
                     "po_sched": 3, "tags": ""})
    df = pd.DataFrame(rows)
    counts = counts or COUNTS
    keep = []
    for pos in POS:
        sub = df[df["pos"] == pos].sort_values(["proj"], ascending=False).head(int(counts.get(pos, 50))).copy()
        sub["tier"] = gap_tiers(sub["proj"].to_numpy())
        sub["floor"] = (sub["proj"] * 0.86).round(1)
        sub["ceil"] = (sub["proj"] * 1.14).round(1)
        keep.append(sub)
    out = pd.concat(keep)[PLAYER_COLS]
    return out, notes


def do_fetch(cfg, season: int | None = None) -> str:
    """Build/refresh players_skeleton.csv. Quick (two API calls); safe to re-run."""
    df, notes = build_skeleton(cfg["data_dir"], season=season)
    path = os.path.join(cfg["data_dir"], "players_skeleton.csv")
    df.to_csv(path, index=False)
    for n in notes:
        print("NOTE:", n)
    print(f"wrote {path}: " + ", ".join(f"{p} {len(df[df.pos==p])}" for p in POS))
    return path


def cmd_fetch(cfg, args):
    do_fetch(cfg, season=getattr(args, "season", None))
    df = pd.read_csv(os.path.join(cfg["data_dir"], "players_skeleton.csv"))
    with pd.option_context("display.width", 160):
        print(df.sort_values("adp").head(12).to_string(index=False))
