"""Load, validate, and resolve the research CSVs against Sleeper's player table."""
from __future__ import annotations

import os
import re
import unicodedata
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

POS = ["QB", "RB", "WR", "TE", "K"]     # modeled positions; leagues without a K slot just never rank one
ROLE = {"LOCKED", "SHARED", "CONTESTED", "UNKNOWN"}
INJ = {"OK", "Q", "D", "O", "PUP", "HOLDOUT", "SUSP"}
TAGS = {"NEW_OC", "NEW_QB", "NEW_TEAM", "ROOKIE", "ROOKIE_THREAT", "HANDCUFF", "RZ_HOG", "YAC", "DEEP",
        "VOLUME", "CONTRACT_YR", "AGE_CLIFF", "IR_STASH", "INJ_HISTORY", "BREAKOUT", "DECLINE"}
STACK_TYPE = {"VOLUME", "DEEP", "RZ", "YAC"}
ANTI_REASON = {"COMMITTEE", "TARGET_SHARE", "RUSH_QB", "GOAL_LINE"}
PLAYER_COLS = ["name", "pos", "team", "bye", "proj", "floor", "ceil", "tier", "adp", "risk", "role", "inj",
               "age", "po_sched", "tags"]
STACK_COLS = ["qb", "rec", "team", "stack_score", "type", "tgt_share", "rz_share", "adot"]
ANTI_COLS = ["p1", "p2", "team", "penalty", "reason"]

_SUFFIX = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b")


def norm_name(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    s = s.lower().replace(".", "").replace("'", "").replace("-", " ")
    s = _SUFFIX.sub("", s)
    return re.sub(r"[^a-z0-9]", "", s)


@dataclass
class Research:
    players: pd.DataFrame
    stacks: pd.DataFrame
    anti: pd.DataFrame
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(p.startswith("ERROR") for p in self.problems)


def _read(path: str, cols: list[str], required: bool = True) -> pd.DataFrame | None:
    if not os.path.exists(path):
        if required:
            raise FileNotFoundError(path)
        return pd.DataFrame(columns=cols)
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df.columns = [c.strip().lower() for c in df.columns]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    for c in df.columns:
        df[c] = df[c].astype(str).str.strip()
    return df


def load_research(data_dir: str, sleeper_players: dict | None = None) -> Research:
    problems: list[str] = []
    pl = _read(os.path.join(data_dir, "players.csv"), PLAYER_COLS)
    if "sleeper_rank" in pl.columns:            # optional: the draft room's own ranking
        pl = pl.rename(columns={"sleeper_rank": "sleeper_rank_override"})
    st = _read(os.path.join(data_dir, "stacks.csv"), STACK_COLS, required=False)
    an = _read(os.path.join(data_dir, "anti.csv"), ANTI_COLS, required=False)

    # ---- types
    for c in ["proj", "floor", "ceil", "adp"]:
        pl[c] = pd.to_numeric(pl[c], errors="coerce")
    for c in ["bye", "tier", "risk", "age", "po_sched"]:
        pl[c] = pd.to_numeric(pl[c], errors="coerce")
    pl["pos"] = pl["pos"].str.upper()
    pl["team"] = pl["team"].str.upper()
    pl["role"] = pl["role"].str.upper()
    pl["inj"] = pl["inj"].str.upper()
    pl["tags"] = pl["tags"].fillna("").str.upper()

    # ---- validation
    bad_pos = pl[~pl["pos"].isin(POS)]
    if len(bad_pos):
        problems.append(f"ERROR players.csv: bad pos rows: {bad_pos['name'].tolist()[:10]}")
    for c in ["proj", "floor", "ceil", "adp", "bye", "tier", "risk", "age", "po_sched"]:
        n = pl[c].isna().sum()
        if n:
            problems.append(f"ERROR players.csv: {n} non-numeric values in '{c}'")
    dup = pl[pl["name"].duplicated()]
    if len(dup):
        problems.append(f"ERROR players.csv: duplicate names {dup['name'].tolist()}")
    fc = pl[~((pl["floor"] < pl["proj"]) & (pl["proj"] < pl["ceil"]))]
    if len(fc):
        problems.append(f"WARN players.csv: floor<proj<ceil violated for {fc['name'].tolist()[:10]}")
    bad_role = pl[~pl["role"].isin(ROLE)]
    if len(bad_role):
        problems.append(f"ERROR players.csv: bad role for {bad_role['name'].tolist()[:10]}")
    bad_inj = pl[~pl["inj"].isin(INJ)]
    if len(bad_inj):
        problems.append(f"ERROR players.csv: bad inj for {bad_inj['name'].tolist()[:10]}")
    for _, r in pl.iterrows():
        for t in [x for x in str(r["tags"]).split("|") if x]:
            if t not in TAGS:
                problems.append(f"WARN players.csv: unknown tag {t!r} on {r['name']}")
    byes = pl[pl["team"] != "FA"].groupby("team")["bye"].nunique()
    multi = byes[byes > 1]
    if len(multi):
        problems.append(f"ERROR players.csv: inconsistent bye within team(s) {multi.index.tolist()}")
    for pos in POS:
        sub = pl[pl["pos"] == pos].sort_values("proj", ascending=False)
        tiers = sub["tier"].dropna().astype(int).tolist()
        if tiers and (tiers[0] != 1 or any(b - a not in (0, 1) for a, b in zip(tiers, tiers[1:]))):
            problems.append(f"WARN players.csv: {pos} tiers not contiguous/monotone in proj order")
    counts = pl["pos"].value_counts().to_dict()
    problems.append(f"INFO players.csv: counts {counts}")

    names = set(pl["name"])
    for c in ["qb", "rec"]:
        miss = st[~st[c].isin(names)][c].tolist()
        if miss:
            problems.append(f"ERROR stacks.csv: unresolved {c}: {miss[:10]}")
    for c in ["p1", "p2"]:
        miss = an[~an[c].isin(names)][c].tolist()
        if miss:
            problems.append(f"ERROR anti.csv: unresolved {c}: {miss[:10]}")
    if len(st):
        st["stack_score"] = pd.to_numeric(st["stack_score"], errors="coerce").fillna(0)
        st["type"] = st["type"].str.upper()
        bad = st[~st["type"].isin(STACK_TYPE)]
        if len(bad):
            problems.append(f"WARN stacks.csv: bad type rows {len(bad)}")
        for c in ["tgt_share", "rz_share", "adot"]:
            st[c] = pd.to_numeric(st[c], errors="coerce")
    if len(an):
        an["penalty"] = pd.to_numeric(an["penalty"], errors="coerce").fillna(0)
        an["reason"] = an["reason"].str.upper()

    # ---- resolve to sleeper ids
    pl["pid"] = ""
    pl["sleeper_rank"] = np.nan
    pl["sleeper_inj"] = ""
    if sleeper_players:
        idx: dict[tuple[str, str], list[dict]] = {}
        idx_name: dict[str, list[dict]] = {}
        for p in sleeper_players.values():
            n = norm_name(p["name"])
            idx.setdefault((n, p["pos"]), []).append(p)
            idx_name.setdefault(n, []).append(p)
        unresolved, ambiguous = [], []
        for i, r in pl.iterrows():
            n = norm_name(r["name"])
            cands = idx.get((n, r["pos"])) or idx_name.get(n) or []
            if len(cands) > 1:
                # prefer team match, then a player with a team, then lowest search_rank
                team_match = [c for c in cands if (c.get("team") or "FA") == r["team"]]
                cands = team_match or [c for c in cands if c.get("team")] or cands
                cands = sorted(cands, key=lambda c: c.get("search_rank") or 10**9)
                if len(team_match) != 1 and len(cands) > 1:
                    ambiguous.append(r["name"])
            if not cands:
                unresolved.append(r["name"])
                continue
            c = cands[0]
            pl.at[i, "pid"] = c["player_id"]
            pl.at[i, "sleeper_rank"] = c.get("search_rank") or np.nan
            pl.at[i, "sleeper_inj"] = c.get("injury_status") or ""
            if (c.get("team") or "FA") != r["team"]:
                problems.append(f"WARN team mismatch {r['name']}: csv={r['team']} sleeper={c.get('team')}")
        if unresolved:
            problems.append(f"ERROR unresolved Sleeper ids (fix spelling): {unresolved}")
        if ambiguous:
            problems.append(f"WARN ambiguous names (picked best guess by team/rank): {ambiguous}")
    else:
        pl["pid"] = ["csv_" + norm_name(n) for n in pl["name"]]
        problems.append("INFO no Sleeper player table: using synthetic ids")

    # autodraft model ranking: optional csv override column, else Sleeper search_rank, else ADP order
    override = pd.to_numeric(pl.get("sleeper_rank_override", pd.Series(index=pl.index, dtype=float)), errors="coerce")
    pl["sleeper_rank"] = override.fillna(pl["sleeper_rank"]).fillna(pl["adp"] * 1000)

    # ---- derived lookups
    name_to_pid = dict(zip(pl["name"], pl["pid"]))
    if len(st):
        st["qb_pid"] = st["qb"].map(name_to_pid)
        st["rec_pid"] = st["rec"].map(name_to_pid)
    if len(an):
        an["p1_pid"] = an["p1"].map(name_to_pid)
        an["p2_pid"] = an["p2"].map(name_to_pid)
    # team's projected starting QB = highest proj QB on team
    qbs = pl[pl["pos"] == "QB"].sort_values("proj", ascending=False).drop_duplicates("team")
    team_qb = dict(zip(qbs["team"], qbs["pid"]))
    pl["team_qb_pid"] = pl["team"].map(team_qb).fillna("")

    pl = pl.set_index("pid", drop=False)
    return Research(players=pl, stacks=st, anti=an, problems=problems)


def tier_table(players: pd.DataFrame, available: set[str] | None = None) -> pd.DataFrame:
    """Per (pos, tier): total, left, top/bottom proj, cliff to next tier."""
    rows = []
    for pos in POS:
        sub = players[players["pos"] == pos]
        tiers = sorted(sub["tier"].dropna().astype(int).unique())
        for t in tiers:
            tt = sub[sub["tier"] == t]
            nxt = sub[sub["tier"] == t + 1]
            cliff = float(tt["proj"].min() - nxt["proj"].max()) if len(nxt) else float("nan")
            left = len(tt) if available is None else int(tt["pid"].isin(available).sum())
            rows.append({"pos": pos, "tier": t, "total": len(tt), "left": left,
                         "top": float(tt["proj"].max()), "bottom": float(tt["proj"].min()), "cliff": cliff})
    return pd.DataFrame(rows)
