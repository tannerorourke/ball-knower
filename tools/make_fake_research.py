"""Generate synthetic players.csv / stacks.csv / anti.csv with the real schema.
Names are fake; only for `draftkit mock --offline` and validating the pipeline."""
import os
import sys

import numpy as np
import pandas as pd

TEAMS = ["ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE", "DAL", "DEN", "DET", "GB", "HOU", "IND", "JAX", "KC",
         "LV", "LAC", "LAR", "MIA", "MIN", "NE", "NO", "NYG", "NYJ", "PHI", "PIT", "SF", "SEA", "TB", "TEN", "WAS"]
COUNTS = {"QB": 24, "RB": 70, "WR": 80, "TE": 24}
TOP = {"QB": 380.0, "RB": 300.0, "WR": 290.0, "TE": 210.0}
DECAY = {"QB": 0.035, "RB": 0.045, "WR": 0.035, "TE": 0.09}
ROLES = ["LOCKED", "SHARED", "CONTESTED", "UNKNOWN"]
TAGS = ["", "", "", "NEW_OC", "ROOKIE", "HANDCUFF", "RZ_HOG", "YAC", "DEEP", "VOLUME", "AGE_CLIFF", "BREAKOUT", "INJ_HISTORY"]


def main(out_dir="data", seed=7):
    rng = np.random.default_rng(seed)
    byes = {t: int(b) for t, b in zip(TEAMS, rng.choice(range(5, 15), size=len(TEAMS)))}
    rows = []
    for pos, n in COUNTS.items():
        proj = TOP[pos] * np.exp(-DECAY[pos] * np.arange(n)) + rng.normal(0, 3, n)
        proj = np.sort(proj)[::-1]
        # tiers by gaps
        gaps = -np.diff(proj)
        thr = np.quantile(gaps, 0.8)
        tier, tiers = 1, []
        for i in range(n):
            if i > 0 and gaps[i - 1] > thr:
                tier += 1
            tiers.append(tier)
        for i in range(n):
            team = TEAMS[(i * 7 + hash(pos) % 5) % len(TEAMS)] if pos != "QB" else TEAMS[i % len(TEAMS)]
            risk = int(np.clip(rng.poisson(1.2), 0, 5))
            spread = 0.15 + 0.05 * risk
            rows.append({
                "name": f"{pos} Player{i+1:02d}", "pos": pos, "team": team, "bye": byes[team],
                "proj": round(float(proj[i]), 1), "floor": round(float(proj[i] * (1 - spread)), 1),
                "ceil": round(float(proj[i] * (1 + spread)), 1), "tier": tiers[i],
                "adp": 0.0, "risk": risk, "role": rng.choice(ROLES, p=[0.6, 0.25, 0.1, 0.05]),
                "inj": rng.choice(["OK", "OK", "OK", "OK", "Q"]), "age": int(rng.integers(22, 33)),
                "po_sched": int(rng.integers(1, 6)), "tags": rng.choice(TAGS),
            })
    pl = pd.DataFrame(rows)
    # ADP: overall order by a value-ish score with positional adjustment + noise
    adjust = {"QB": -110, "RB": 0, "WR": 0, "TE": -55}
    score = pl["proj"] + pl["pos"].map(adjust) + rng.normal(0, 6, len(pl))
    pl["adp"] = score.rank(ascending=False).astype(float)
    pl.loc[pl["adp"] > 190, "adp"] = 999.0
    pl = pl.sort_values(["pos", "proj"], ascending=[True, False], key=lambda s: s.map({"QB": 0, "RB": 1, "WR": 2, "TE": 3}) if s.name == "pos" else s)
    os.makedirs(out_dir, exist_ok=True)
    pl.to_csv(os.path.join(out_dir, "players.csv"), index=False)

    # stacks: each QB with the top 2 WR/TE/RB on his team
    st = []
    for _, qb in pl[pl.pos == "QB"].iterrows():
        recs = pl[(pl.team == qb.team) & (pl.pos != "QB")].sort_values("proj", ascending=False).head(3)
        for _, r in recs.iterrows():
            st.append({"qb": qb["name"], "rec": r["name"], "team": qb.team,
                       "stack_score": int(rng.integers(30, 95)), "type": rng.choice(["VOLUME", "DEEP", "RZ", "YAC"]),
                       "tgt_share": round(float(rng.uniform(0.1, 0.3)), 2), "rz_share": round(float(rng.uniform(0.1, 0.35)), 2),
                       "adot": round(float(rng.uniform(1, 14)), 1)})
    pd.DataFrame(st).to_csv(os.path.join(out_dir, "stacks.csv"), index=False)

    an = []
    for team in TEAMS:
        rbs = pl[(pl.team == team) & (pl.pos == "RB")].sort_values("proj", ascending=False)
        if len(rbs) >= 2:
            an.append({"p1": rbs.iloc[0]["name"], "p2": rbs.iloc[1]["name"], "team": team,
                       "penalty": round(float(rng.uniform(4, 10)), 1), "reason": "COMMITTEE"})
    pd.DataFrame(an).to_csv(os.path.join(out_dir, "anti.csv"), index=False)
    print(f"wrote fake research to {out_dir}/ ({len(pl)} players, {len(st)} stacks, {len(an)} anti)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data")
