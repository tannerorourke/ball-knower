"""Replacement level, VORP, roster fit, and the deterministic candidate ranking."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .data import POS

FLEX_DEFS = {
    "FLEX": {"RB", "WR", "TE"},
    "SUPER_FLEX": {"QB", "RB", "WR", "TE"},
    "REC_FLEX": {"WR", "TE"},
    "WRRB_FLEX": {"RB", "WR"},
}


@dataclass
class Slots:
    fixed: dict[str, int]                       # QB/RB/WR/TE fixed starter slots per team
    flex: list[set[str]]                        # one entry per flex slot, eligible positions
    bench: int
    ir: int
    ignored: list[str] = field(default_factory=list)

    @classmethod
    def from_roster_positions(cls, rp: list[str]) -> "Slots":
        fixed = {p: 0 for p in POS}
        flex, bench, ir, ignored = [], 0, 0, []
        for s in rp:
            if s in fixed:
                fixed[s] += 1
            elif s in FLEX_DEFS:
                flex.append(set(FLEX_DEFS[s]))
            elif s == "BN":
                bench += 1
            elif s == "IR":
                ir += 1
            else:
                ignored.append(s)     # DEF etc. - not modeled, but still consumes a draft round
        return cls(fixed=fixed, flex=flex, bench=bench, ir=ir, ignored=ignored)

    @property
    def starters(self) -> int:
        return sum(self.fixed.values()) + len(self.flex)

    @property
    def ignored_startable(self) -> int:
        """Startable slots we don't model (DEF): each still eats a late round."""
        return len(self.ignored)


def replacement_levels(players: pd.DataFrame, teams: int, slots: Slots, value_col: str = "proj",
                       extra: dict[str, int] | None = None) -> dict[str, float]:
    """League-wide greedy starter fill. Replacement = best player left at each
    position after every starting slot (fixed then flex) is filled, shifted by
    `extra` more players to reflect that benches soak up a few starters' worth."""
    extra = extra or {}
    avail = {p: players[players["pos"] == p].sort_values(value_col, ascending=False)[value_col].tolist() for p in POS}
    ptr = {p: 0 for p in POS}
    for p in POS:
        ptr[p] = min(len(avail[p]), teams * slots.fixed[p])
    for elig in slots.flex:
        for _ in range(teams):
            best, bp = -1e9, None
            for p in elig:
                if ptr[p] < len(avail[p]) and avail[p][ptr[p]] > best:
                    best, bp = avail[p][ptr[p]], p
            if bp:
                ptr[bp] += 1
    repl = {}
    for p in POS:
        i = min(len(avail[p]) - 1, ptr[p] + int(extra.get(p, 0)))
        repl[p] = float(avail[p][i]) if avail[p] else 0.0
    return repl


def ceiling_gamma(rnd: int, blend: dict[int, float]) -> float:
    """Piecewise-constant weight on (ceil - proj) by round."""
    g = 0.0
    for k in sorted(blend):
        if rnd >= int(k):
            g = float(blend[k])
    return g


def value_column(players: pd.DataFrame, rnd: int, blend: dict[int, float]) -> pd.Series:
    g = ceiling_gamma(rnd, blend)
    return players["proj"] + g * (players["ceil"] - players["proj"])


# ---------------------------------------------------------------------------
# My roster / needs
# ---------------------------------------------------------------------------
def roster_fill(roster: pd.DataFrame, slots: Slots) -> tuple[dict[str, int], int, list[str]]:
    """Greedy assignment of my roster to starting slots by proj.
    Returns (fixed_open per pos, flex_open, bench pids)."""
    fixed_open = dict(slots.fixed)
    flex_open = len(slots.flex)
    bench = []
    for pid, r in roster.sort_values("proj", ascending=False).iterrows():
        p = r["pos"]
        if fixed_open.get(p, 0) > 0:
            fixed_open[p] -= 1
        elif flex_open > 0 and any(p in e for e in slots.flex):
            flex_open -= 1
        else:
            bench.append(pid)
    return fixed_open, flex_open, bench


def starters_by_slot(roster: pd.DataFrame, slots: Slots) -> dict[str, list[str]]:
    """Which pids occupy fixed slots per position (for bye collisions)."""
    out = {p: [] for p in POS}
    fixed_open = dict(slots.fixed)
    for pid, r in roster.sort_values("proj", ascending=False).iterrows():
        p = r["pos"]
        if fixed_open.get(p, 0) > 0:
            fixed_open[p] -= 1
            out[p].append(pid)
    return out


def bye_cost(cand: pd.Series, roster: pd.DataFrame, slots: Slots, per_collision: float) -> tuple[float, list[str]]:
    """Cost of the candidate sharing a bye with players already occupying the
    same position's starting slots. Only counts when the fixed slots at that
    position would all be dark the same week."""
    if roster.empty:
        return 0.0, []
    same = roster[(roster["pos"] == cand["pos"]) & (roster["bye"] == cand["bye"])]
    n_fixed = slots.fixed.get(cand["pos"], 0)
    if len(same) == 0:
        return 0.0, []
    # collision if adding cand means >= n_fixed starters at this pos share the bye
    if len(same) + 1 >= max(1, n_fixed):
        return per_collision * len(same), same["name"].tolist()
    return per_collision * 0.35 * len(same), same["name"].tolist()   # partial: flex-level exposure


def stack_bonus(cand: pd.Series, roster: pd.DataFrame, stacks: pd.DataFrame, max_bonus: float) -> tuple[float, str]:
    if stacks is None or stacks.empty or roster.empty:
        return 0.0, ""
    mine = set(roster.index)
    if cand["pos"] == "QB":
        rows = stacks[(stacks["qb_pid"] == cand.name) & (stacks["rec_pid"].isin(mine))]
    else:
        rows = stacks[(stacks["rec_pid"] == cand.name) & (stacks["qb_pid"].isin(mine))]
    if rows.empty:
        return 0.0, ""
    best = rows.sort_values("stack_score", ascending=False).iloc[0]
    partner = best["qb"] if cand["pos"] != "QB" else best["rec"]
    return max_bonus * float(best["stack_score"]) / 100.0, f"{partner} ({best['type']})"


def anti_penalty(cand: pd.Series, roster: pd.DataFrame, anti: pd.DataFrame) -> tuple[float, str]:
    if anti is None or anti.empty or roster.empty:
        return 0.0, ""
    mine = set(roster.index)
    rows = anti[((anti["p1_pid"] == cand.name) & (anti["p2_pid"].isin(mine))) |
                ((anti["p2_pid"] == cand.name) & (anti["p1_pid"].isin(mine)))]
    if rows.empty:
        return 0.0, ""
    pen = float(rows["penalty"].sum())
    who = ", ".join((rows["p2"].where(rows["p1_pid"] == cand.name, rows["p1"])).tolist())
    return pen, who


def roster_needs(roster: pd.DataFrame, slots: Slots, cfg: dict) -> dict:
    """Once-per-tick roster context: open starters, per-position backup deficits,
    and same-position byes (for backup diversity)."""
    counts = roster["pos"].value_counts().to_dict() if len(roster) else {}
    if len(roster):
        fixed_open, flex_open, bench_pids = roster_fill(roster, slots)
    else:
        fixed_open, flex_open, bench_pids = dict(slots.fixed), len(slots.flex), []
    targets = cfg.get("backup_targets", {})
    bench_pos = roster.loc[bench_pids]["pos"].value_counts().to_dict() if bench_pids else {}
    backup_open = {}
    for p in POS:
        if slots.fixed.get(p, 0) == 0 and not any(p in e for e in slots.flex):
            continue                                    # position not startable in this league
        tgt = int(targets.get(p, min(slots.fixed.get(p, 0), 2)))
        backup_open[p] = max(0, tgt - bench_pos.get(p, 0))
    pos_byes = {p: set(roster.loc[roster["pos"] == p, "bye"].astype(int)) for p in POS} if len(roster) else {p: set() for p in POS}
    return {
        "counts": counts, "fixed_open": fixed_open, "flex_open": flex_open,
        "bench_pos": bench_pos, "backup_open": backup_open, "pos_byes": pos_byes,
        # the gate never counts K: a missing kicker must not block depth picks
        "starters_open_gate": sum(n for p, n in fixed_open.items() if p != "K") + flex_open,
        "backups_open_total": sum(backup_open.values()),
    }


def need_bonus(pos: str, bye: int, ctx: dict, slots: Slots, rnd: int, rounds: int, cfg: dict) -> tuple[float, str, str]:
    """Roster-phase classifier: fill starters, then direct backups, then maximize.
    Returns (points, tag, note). Tags: STARTER FLEX BACKUP DEPTH CAPPED K-WAIT."""
    w = float(cfg.get("need_weight", 6.0))
    caps = cfg.get("pos_caps", {})
    bp = cfg.get("bench_penalty", {"QB": 12.0, "TE": 9.0, "RB": 2.0, "WR": 2.0, "K": 15.0})
    if ctx["counts"].get(pos, 0) >= int(caps.get(pos, 99)):
        return -float(cfg.get("capped_multiplier", 3.0)) * w, "CAPPED", ""
    fixed_open, flex_open = ctx["fixed_open"], ctx["flex_open"]
    # DEF etc. still eat a round each, so the endgame squeeze comes one round early per ignored slot
    rounds_left = rounds - rnd + 1 - slots.ignored_startable
    open_all = sum(fixed_open.values()) + flex_open
    endgame = rounds_left <= open_all + 1
    urgency = float(cfg.get("urgency_factor", 2.0)) if endgame else 1.0
    if pos == "K" and fixed_open.get("K", 0) > 0 and not endgame:
        return -float(bp.get("K", 15.0)), "K-WAIT", "kicker waits for the endgame"
    if fixed_open.get(pos, 0) > 0:
        return w * urgency, "STARTER", ""
    if flex_open > 0 and any(pos in e for e in slots.flex):
        return 0.5 * w * urgency, "FLEX", ""
    # the backup phase only opens once every (non-K) starter is filled -
    # until then a would-be backup is just depth and falls through to the gate
    if ctx["backup_open"].get(pos, 0) > 0 and ctx["starters_open_gate"] == 0:
        pts = float(cfg.get("backup_weight", 0.8)) * w * urgency
        # a backup only covers the starter if their byes differ
        if int(bye) in ctx["pos_byes"].get(pos, set()):
            pts *= float(cfg.get("backup_bye_overlap_factor", 0.4))
            return pts, "BACKUP", f"bye {int(bye)} overlaps my {pos}s - weak cover"
        return pts, "BACKUP", ""
    # DEPTH: everything past starters + backup targets. Heavily gated while
    # starters (non-K) are open, lightly gated while backup targets are open.
    depth = ctx["bench_pos"].get(pos, 0) + 1
    late = rnd >= int(cfg.get("bench_free_round", 9))
    pen = float(bp.get(pos, 2.0)) * depth * (0.5 if late else 1.0)
    note = ""
    if ctx["starters_open_gate"] > 0:
        pen += float(cfg.get("depth_while_starters_open", 15.0))
        note = "starters still open"
    elif ctx["backups_open_total"] > 0:
        pen += float(cfg.get("depth_while_backups_open", 6.0))
        note = "backup slots still open"
    return -pen, "DEPTH", note


# ---------------------------------------------------------------------------
# Candidate table
# ---------------------------------------------------------------------------
def candidate_table(players: pd.DataFrame, taken: set[str], my_pids: list[str], rnd: int, rounds: int,
                    slots: Slots, teams: int, mc: dict | None, stacks: pd.DataFrame, anti: pd.DataFrame,
                    cfg: dict, top_n: int = 8) -> pd.DataFrame:
    """Deterministic ranking of available players for my current pick."""
    blend = {int(k): float(v) for k, v in cfg.get("ceiling_blend", {1: 0.0}).items()}
    val = value_column(players, rnd, blend)
    df = players.assign(val=val)
    repl = replacement_levels(df, teams, slots, value_col="val", extra=cfg.get("replacement_extra", {}))
    df["vorp"] = df["val"] - df["pos"].map(repl)
    avail = df[~df.index.isin(taken)].copy()
    roster = players.loc[[p for p in my_pids if p in players.index]]

    surv = (mc or {}).get("surv", {})
    e_next = (mc or {}).get("pos_expect", {})
    tier_empty = (mc or {}).get("tier_empty", {})
    # A player headed for a flex or bench slot competes with every flex-eligible
    # position, so his value is measured against the flex replacement level,
    # not his own position's. (TE vorp over TE12 overstates a TE in the flex.)
    flex_elig = set().union(*slots.flex) if slots.flex else set()
    flex_repl = max([repl[p] for p in flex_elig], default=0.0)
    ctx = roster_needs(roster, slots, cfg)
    wait_w_map = {"STARTER": 1.0, "FLEX": 0.8, "BACKUP": 0.6, "DEPTH": 0.15, "CAPPED": 0.0, "K-WAIT": 0.0}
    wait_w_map.update({k: float(v) for k, v in (cfg.get("wait_w") or {}).items()})
    roster_byes = roster["bye"].astype(int) if len(roster) else pd.Series(dtype=int)

    rows = []
    for pid, r in avail.iterrows():
        pos = r["pos"]
        wait_cost = float(r["vorp"] - e_next.get(pos, r["vorp"]))     # what I lose by waiting a turn
        nb, need_tag, need_note = need_bonus(pos, int(r["bye"]), ctx, slots, rnd, rounds, cfg)
        # waiting only costs something if I actually intend to fill the slot
        wait_w = wait_w_map.get(need_tag, 0.0)
        vorp_eff = float(r["vorp"])
        if need_tag in ("FLEX", "BACKUP", "DEPTH") and pos in flex_elig:
            vorp_eff -= max(0.0, flex_repl - repl[pos])
        # bye collisions are draft-dependent: displayed and warned about, never scored
        bc, bye_with = bye_cost(r, roster, slots, float(cfg.get("bye_cost_per_collision", 4.0)))
        bye_team_count = int((roster_byes == int(r["bye"])).sum())
        sb, stack_with = stack_bonus(r, roster, stacks, float(cfg.get("stack_max_bonus", 6.0)))
        ap, anti_with = anti_penalty(r, roster, anti)
        risk_pen = float(cfg.get("risk_weight", 1.0)) * float(r["risk"]) * (1.0 if rnd <= 6 else 0.5)
        adj = wait_w * max(wait_cost, 0.0) + float(cfg.get("vorp_weight", 0.15)) * vorp_eff + nb + sb - ap - risk_pen
        rows.append({
            "pid": pid, "name": r["name"], "pos": pos, "team": r["team"], "bye": int(r["bye"]),
            "tier": int(r["tier"]), "proj": round(float(r["proj"]), 1), "vorp": round(float(r["vorp"]), 1),
            "adp": float(r["adp"]), "surv": float(surv.get(pid, 1.0)),
            "wait_cost": round(wait_cost, 1), "need": need_tag, "need_pts": round(nb, 1), "need_note": need_note,
            "bye_cost": round(bc, 1), "bye_with": ", ".join(bye_with), "bye_team_count": bye_team_count,
            "stack": round(sb, 1), "stack_with": stack_with, "anti": round(ap, 1), "anti_with": anti_with,
            "risk": int(r["risk"]), "role": r["role"], "inj": r["inj"], "tags": r["tags"],
            "tier_empty_p": float(tier_empty.get((pos, int(r["tier"])), 0.0)),
            "adj": round(adj, 1),
        })
    out = pd.DataFrame(rows).sort_values(["adj", "vorp"], ascending=False)
    out.attrs["replacement"] = repl
    out.attrs["fixed_open"] = ctx["fixed_open"]
    out.attrs["flex_open"] = ctx["flex_open"]
    out.attrs["backup_open"] = ctx["backup_open"]
    out.attrs["e_next"] = e_next
    return out.head(top_n) if top_n else out
