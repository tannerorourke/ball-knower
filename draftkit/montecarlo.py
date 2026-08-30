"""Monte Carlo survival: who is still on the board at my next pick?

Each sim replays the opponent picks between now and the horizon pick.
Autodraft teams pick deterministically (Sleeper rank, need-filtered).
Human teams draft from a noisy personal board:
    score = adp + sigma(adp) * (eps + need_shift(pos) - run_shift * recent_same_pos)
and take the lowest score. sigma(adp) = sigma_base + sigma_slope*adp is the ADP
standard deviation (published per player; grows with pick number). Need and run
shifts are therefore expressed in standard deviations, so they mean the same
thing at pick 5 and pick 95.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import POS
from .scoring import Slots, roster_fill

POS_IDX = {p: i for i, p in enumerate(POS)}


def _need_shift(pos: str, roster: pd.DataFrame, slots: Slots, rounds_left: int, caps: dict, shift: dict) -> float:
    """Additive shift (in ADP standard deviations) applied to a team's personal
    board. 0 = no adjustment; positive pushes the position down that board."""
    counts = roster["pos"].value_counts().to_dict() if len(roster) else {}
    if counts.get(pos, 0) >= int(caps.get(pos, 99)):
        return float(shift.get("CAPPED", 20))
    fixed_open, flex_open, _ = roster_fill(roster, slots) if len(roster) else (dict(slots.fixed), len(slots.flex), [])
    open_starters = sum(fixed_open.values()) + flex_open
    urgent = rounds_left <= open_starters + 1
    if fixed_open.get(pos, 0) > 0:
        return -float(shift.get("URGENT", 1.0)) if urgent else float(shift.get("STARTER", 0))
    if flex_open > 0 and any(pos in e for e in slots.flex):
        return float(shift.get("FLEX", 0.5))
    return float(shift.get("BENCH", 1.5)) + (float(shift.get("URGENT", 1.0)) if urgent else 0.0)


def simulate(players: pd.DataFrame, taken: set[str], upcoming_slots: list[int], rosters: dict[int, list[str]],
             autodraft: set[int], slots: Slots, rnd: int, rounds: int, recent_pos: list[str], cfg: dict,
             rng: np.random.Generator | None = None, vorp: pd.Series | None = None) -> dict:
    """Returns {'surv': {pid: p}, 'pos_expect': {pos: E[max vorp]}, 'tier_empty': {(pos,tier): p}, 'n_sims': S}."""
    rng = rng or np.random.default_rng()
    S = int(cfg.get("sims", 2000))
    # pool must cover every pick in the window plus a margin, else sims run dry
    pool_n = max(int(cfg.get("pool", 80)), len(upcoming_slots) + 40)
    sigma_base = float(cfg.get("sigma_base", 1.0))
    sigma_slope = float(cfg.get("sigma_slope", 0.12))
    run_shift = float(cfg.get("run_shift", 0.4))
    k = int(cfg.get("run_window", 4))
    caps = cfg.get("pos_caps", {"QB": 2, "TE": 2, "RB": 7, "WR": 7})
    shift = cfg.get("need_shift", {})

    avail_df = players[~players.index.isin(taken)].sort_values("adp")
    if vorp is None:
        vorp = players["proj"] - players.groupby("pos")["proj"].transform(lambda s: s.nlargest(30).min())
    if not upcoming_slots or avail_df.empty:
        return {"surv": {pid: 1.0 for pid in avail_df.index}, "pos_expect": {}, "tier_empty": {}, "n_sims": 0}

    pool = avail_df.head(pool_n)
    pids = pool.index.to_numpy()
    N = len(pids)
    pos_code = pool["pos"].map(POS_IDX).to_numpy()
    tier = pool["tier"].fillna(99).astype(int).to_numpy()
    v = vorp.reindex(pids).fillna(0).to_numpy(dtype=float)
    adp = pool["adp"].fillna(999).to_numpy(dtype=float)
    adp = np.where(adp >= 999, 200 + np.arange(N), adp)          # undrafted: deep on every board
    sigma = sigma_base + sigma_slope * adp
    s_rank_order = np.argsort(pool["sleeper_rank"].fillna(1e9).to_numpy())

    avail = np.ones((S, N), dtype=bool)
    hist = np.full((S, k), -1, dtype=int)
    for i, p in enumerate(recent_pos[-k:]):
        hist[:, k - len(recent_pos[-k:]) + i] = POS_IDX.get(p, -1)

    rounds_left = rounds - rnd + 1
    row = np.arange(S)
    for slot in upcoming_slots:
        roster = players.loc[[p for p in rosters.get(slot, []) if p in players.index]]
        ns = np.array([_need_shift(POS[c], roster, slots, rounds_left, caps, shift) for c in range(4)])[pos_code]
        if slot in autodraft:
            # best sleeper-ranked available that isn't capped for this roster
            ok = avail[:, s_rank_order] & (ns[s_rank_order] < 10)[None, :]
            first = np.argmax(ok, axis=1)
            has = ok[row, first]
            idx = s_rank_order[first]
            idx = np.where(has, idx, np.argmax(avail, axis=1))
        else:
            run_ct = np.stack([(hist == c).sum(axis=1) for c in range(4)], axis=1)   # S x 4
            score = adp[None, :] + sigma[None, :] * (rng.standard_normal((S, N)) + ns[None, :]
                                                     - run_shift * run_ct[:, pos_code])
            score = np.where(avail, score, np.inf)
            idx = np.argmin(score, axis=1)
        avail[row, idx] = False
        hist = np.roll(hist, -1, axis=1)
        hist[:, -1] = pos_code[idx]

    surv = avail.mean(axis=0)
    out_surv = {pid: float(s) for pid, s in zip(pids, surv)}
    for pid in avail_df.index[pool_n:]:
        out_surv[pid] = 1.0

    pos_expect = {}
    tier_empty = {}
    for c, p in enumerate(POS):
        m = pos_code == c
        if not m.any():
            # nobody in pool at this pos: fall back to best available outside pool
            rest = avail_df[avail_df["pos"] == p]
            pos_expect[p] = float(vorp.reindex(rest.index).max()) if len(rest) else 0.0
            continue
        vv = np.where(avail[:, m], v[m][None, :], -1e9)
        best = vv.max(axis=1)
        # if a sim has nothing left at this pos inside the pool, use best outside pool
        rest = avail_df[(avail_df["pos"] == p) & ~avail_df.index.isin(pids)]
        fallback = float(vorp.reindex(rest.index).max()) if len(rest) else float(v[m].min())
        best = np.where(best < -1e8, fallback, best)
        pos_expect[p] = float(best.mean())
        for t in np.unique(tier[m]):
            mt = m & (tier == t)
            tier_empty[(p, int(t))] = float((~avail[:, mt]).all(axis=1).mean())
    return {"surv": out_surv, "pos_expect": pos_expect, "tier_empty": tier_empty, "n_sims": S}


def sample_one_pick(players: pd.DataFrame, taken: set[str], slot: int, rosters: dict[int, list[str]],
                    autodraft: set[int], slots: Slots, rnd: int, rounds: int, recent_pos: list[str],
                    cfg: dict, rng: np.random.Generator) -> str:
    """Single-draw opponent pick — used by mock mode to drive a fake draft."""
    res_cfg = dict(cfg)
    res_cfg["sims"] = 1
    avail_df = players[~players.index.isin(taken)].sort_values("adp")
    pool = avail_df.head(int(cfg.get("pool", 80)))
    r = simulate(players, taken, [slot], rosters, autodraft, slots, rnd, rounds, recent_pos, res_cfg, rng)
    gone = [pid for pid in pool.index if r["surv"].get(pid, 1.0) < 0.5]
    return gone[0] if gone else pool.index[0]
