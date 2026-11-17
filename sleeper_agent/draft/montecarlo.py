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

from ..core.data import POS
from ..core.scoring import Slots, roster_fill

POS_IDX = {p: i for i, p in enumerate(POS)}


def slot_tendencies(picks: list[dict], players: pd.DataFrame, cfg: dict) -> dict[int, dict]:
    """Learned per-slot drafting style from the picks already made.

    sig_mult: how ADP-anchored a slot is - rms of (adp - pick_no) vs the rms the
    sigma model expects at their pick numbers. Reachers get a wider board, chalk
    drafters a tighter one. A uniform reach bias cancels in the argmin, so bias
    is expressed as spread, not location.
    pos_shift: positional lean vs the room (slot's pick share minus everyone's),
    in sigma units; negative pulls that position up their board.
    Both shrink toward neutral with few samples (tendency_prior_picks)."""
    if not cfg.get("slot_tendencies", True) or not picks:
        return {}
    prior = float(cfg.get("tendency_prior_picks", 4))
    lean_scale = float(cfg.get("lean_shift", 1.0))
    sb = float(cfg.get("sigma_base", 1.0))
    ss = float(cfg.get("sigma_slope", 0.12))
    adp, pos = players["adp"], players["pos"]
    rows = []
    for p in picks:
        pid = str(p["player_id"])
        a = adp.get(pid)
        if a is None or not (float(a) < 900):        # off-board or undrafted-ADP: no signal
            continue
        rows.append((int(p.get("draft_slot") or 0), int(p["pick_no"]), float(a), pos.get(pid)))
    if not rows:
        return {}
    df = pd.DataFrame(rows, columns=["slot", "pick_no", "adp", "pos"])
    league_share = df["pos"].value_counts(normalize=True)
    out = {}
    for slot, g in df.groupby("slot"):
        n = len(g)
        delta = g["adp"] - g["pick_no"]
        exp_rms = float(np.sqrt(((sb + ss * g["pick_no"]) ** 2).mean()))
        raw = float(np.sqrt((delta ** 2).mean())) / max(exp_rms, 1e-6)
        raw = float(np.clip(raw, 1 / 3, 3.0))       # bound before shrinking, or one wild reach dominates
        mult = float(np.clip((prior * 1.0 + n * raw) / (prior + n), 0.5, 2.5))
        w = n / (n + prior)
        shift = {}
        for ps, share in g["pos"].value_counts(normalize=True).items():
            lean = float(share - league_share.get(ps, 0.0))
            if abs(lean) > 1e-9:
                shift[ps] = round(-lean_scale * lean * w, 3)
        out[int(slot)] = {"sig_mult": round(mult, 2), "pos_shift": shift, "n": n}
    return out


def _need_shift(pos: str, roster: pd.DataFrame, slots: Slots, rounds_left: int, caps: dict, shift: dict) -> float:
    """Additive shift (in ADP standard deviations) applied to a team's personal
    board. 0 = no adjustment; positive pushes the position down that board."""
    counts = roster["pos"].value_counts().to_dict() if len(roster) else {}
    if counts.get(pos, 0) >= int(caps.get(pos, 99)):
        return float(shift.get("CAPPED", 20))
    if len(roster):
        fixed_open, flex_open, bench_pids = roster_fill(roster, slots)
    else:
        fixed_open, flex_open, bench_pids = dict(slots.fixed), len(slots.flex), []
    open_starters = sum(fixed_open.values()) + flex_open
    urgent = rounds_left <= open_starters + 1
    if pos == "K" and not urgent:
        return float(shift.get("CAPPED", 20)) / 2     # nobody reaches for a kicker mid-draft
    if fixed_open.get(pos, 0) > 0:
        return -float(shift.get("URGENT", 1.0)) if urgent else float(shift.get("STARTER", 0))
    if flex_open > 0 and any(pos in e for e in slots.flex):
        return float(shift.get("FLEX", 0.5))
    # opponents roster ~1 backup per starter (onesies: 1) before going pure depth
    bench_ct = int((roster.loc[bench_pids, "pos"] == pos).sum()) if bench_pids else 0
    tgt = 1 if slots.fixed.get(pos, 0) <= 1 else 2
    if (slots.fixed.get(pos, 0) or any(pos in e for e in slots.flex)) and bench_ct < tgt:
        return float(shift.get("BACKUP", 1.0))
    return float(shift.get("DEPTH", shift.get("BENCH", 1.5))) + (float(shift.get("URGENT", 1.0)) if urgent else 0.0)


def simulate(players: pd.DataFrame, taken: set[str], upcoming_slots: list[int], rosters: dict[int, list[str]],
             autodraft: set[int], slots: Slots, rnd: int, rounds: int, recent_pos: list[str], cfg: dict,
             rng: np.random.Generator | None = None, vorp: pd.Series | None = None,
             tendencies: dict[int, dict] | None = None) -> dict:
    """Returns {'surv': {pid: p}, 'pos_expect': {pos: E[max vorp]}, 'tier_empty': {(pos,tier): p}, 'n_sims': S}."""
    rng = rng or np.random.default_rng()
    S = int(cfg.get("sims", 2000))
    # pool must cover every pick in the window plus a margin, else sims run dry
    pool_n = max(int(cfg.get("pool", 80)), len(upcoming_slots) + 40)
    sigma_base = float(cfg.get("sigma_base", 1.0))
    sigma_slope = float(cfg.get("sigma_slope", 0.12))
    run_shift = float(cfg.get("run_shift", 0.4))
    k = int(cfg.get("run_window", 4))
    caps = cfg.get("pos_caps", {"QB": 2, "TE": 2, "RB": 7, "WR": 7, "K": 1})
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
        ns = np.array([_need_shift(POS[c], roster, slots, rounds_left, caps, shift) for c in range(len(POS))])[pos_code]
        if slot in autodraft:
            # best sleeper-ranked available that isn't capped for this roster
            ok = avail[:, s_rank_order] & (ns[s_rank_order] < 10)[None, :]
            first = np.argmax(ok, axis=1)
            has = ok[row, first]
            idx = s_rank_order[first]
            idx = np.where(has, idx, np.argmax(avail, axis=1))
        else:
            run_ct = np.stack([(hist == c).sum(axis=1) for c in range(len(POS))], axis=1)   # S x n_pos
            t = (tendencies or {}).get(slot) or {}
            mult = float(t.get("sig_mult", 1.0))
            tshift = t.get("pos_shift") or {}
            ts = np.array([float(tshift.get(p, 0.0)) for p in POS])[pos_code] if tshift else np.zeros(N)
            score = adp[None, :] + sigma[None, :] * (rng.standard_normal((S, N)) * mult + ns[None, :]
                                                     + ts[None, :] - run_shift * run_ct[:, pos_code])
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
    """Single-draw opponent pick - used by mock mode to drive a fake draft."""
    res_cfg = dict(cfg)
    res_cfg["sims"] = 1
    avail_df = players[~players.index.isin(taken)].sort_values("adp")
    pool = avail_df.head(int(cfg.get("pool", 80)))
    r = simulate(players, taken, [slot], rosters, autodraft, slots, rnd, rounds, recent_pos, res_cfg, rng)
    gone = [pid for pid in pool.index if r["surv"].get(pid, 1.0) < 0.5]
    return gone[0] if gone else pool.index[0]
