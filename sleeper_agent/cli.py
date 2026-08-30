"""sleeper-agent command line: setup | validate | script | run | mock"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import yaml

from .data import load_research, tier_table
from .fetch import cmd_fetch, do_fetch
from .research import cmd_research
from .engine import Engine
from .montecarlo import simulate
from .scoring import Slots, replacement_levels
from .sleeper import Sleeper
from .state import DraftMeta, DraftState, my_pick_numbers, round_and_pick

DEFAULTS = {
    "league_id": "", "draft_id": "", "username": "", "my_slot": None,
    "data_dir": "data", "out_dir": "out", "poll_interval": 2.0, "top_n": 12, "seed": 0,
    "mc": {"sims": 2000, "pool": 80, "sigma_base": 1.0, "sigma_slope": 0.12, "run_shift": 0.4, "run_window": 4,
           "need_shift": {"STARTER": 0, "FLEX": 0.5, "BACKUP": 1.0, "DEPTH": 1.5, "URGENT": 1.0, "CAPPED": 20},
           "autodraft_latency_s": 3.0, "autodraft_slots": [], "pos_caps": {"QB": 2, "TE": 2, "RB": 7, "WR": 7, "K": 1},
           "slot_tendencies": True, "tendency_prior_picks": 4, "lean_shift": 1.0},
    "scoring": {"ceiling_blend": {1: 0.0, 6: 0.15, 9: 0.3, 11: 0.5}, "need_weight": 6.0,
                "backup_weight": 0.8, "backup_targets": {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "K": 0},
                "depth_while_starters_open": 15.0, "depth_while_backups_open": 6.0,
                "backup_bye_overlap_factor": 0.4,
                "wait_w": {"STARTER": 1.0, "FLEX": 0.8, "BACKUP": 0.6, "DEPTH": 0.15, "CAPPED": 0.0, "K-WAIT": 0.0},
                "capped_multiplier": 3.0, "urgency_factor": 2.0,
                "stack_max_bonus": 6.0, "risk_weight": 1.0,
                "replacement_extra": {"QB": 3, "RB": 2, "WR": 2, "TE": 2, "K": 1}, "vorp_weight": 0.15,
                "bench_penalty": {"QB": 12.0, "TE": 9.0, "RB": 2.0, "WR": 2.0, "K": 15.0}, "bench_free_round": 9,
                "pos_caps": {"QB": 3, "TE": 2, "RB": 7, "WR": 7, "K": 1}},
    "llm": {"backend": "none", "research_model": "", "draft_model": "",
            "base_url": "", "api_key_env": "LLM_API_KEY",
            "research_timeout_s": 900, "draft_timeout_s": 90,
            "at_n_minus": False, "n_minus": 2, "eta_trigger_s": 60, "post_pick": True},
    "strategy_notes": "",
    "mock": {"autodraft_teams": 3, "autodraft_slots": [], "human_min_s": 8, "human_max_s": 45,
             "my_pick_delay_s": 20, "speed": 10,
             "teams": 12, "rounds": 14, "my_slot": 5, "pick_timer": 180,
             "fault_at_pick": 0, "fault_n_polls": 0,
             "roster_positions": ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "FLEX", "BN", "BN", "BN", "BN", "BN", "BN", "IR"]},
}


def _merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in (b or {}).items():
        out[k] = _merge(a[k], v) if isinstance(v, dict) and isinstance(a.get(k), dict) else v
    return out


def load_config(path: str) -> dict:
    cfg = {}
    if os.path.exists(path):
        with open(path) as f:
            cfg = yaml.safe_load(f) or {}
    return _merge(DEFAULTS, cfg)


# ---------------------------------------------------------------------------
def resolve_draft(cfg: dict, api: Sleeper) -> tuple[dict, dict, int, dict[int, str]]:
    """-> (league, draft, my_slot, slot_names)"""
    league = api.league(cfg["league_id"]) if cfg.get("league_id") else None
    draft_id = cfg.get("draft_id")
    if not draft_id:
        drafts = api.league_drafts(cfg["league_id"])
        if not drafts:
            sys.exit("no drafts on this league yet")
        draft_id = drafts[0]["draft_id"]
    draft = api.draft(draft_id)
    if league is None and draft.get("league_id"):
        league = api.league(draft["league_id"])
    users = api.league_users(league["league_id"]) if league else []
    uname = {u["user_id"]: (u.get("display_name") or u["user_id"]) for u in users}
    order = {str(k): int(v) for k, v in (draft.get("draft_order") or {}).items()}
    slot_names = {slot: uname.get(uid, uid) for uid, slot in order.items()}
    my_slot = cfg.get("my_slot")
    if not my_slot and cfg.get("username"):
        me = api.user(cfg["username"])
        my_slot = order.get(str(me["user_id"]))
    if not my_slot:
        sys.exit("cannot determine my_slot: set username (once draft order is posted) or my_slot in config.yaml")
    return league, draft, int(my_slot), slot_names


def meta_from_cfg(cfg: dict, api: Sleeper | None):
    if api is None:  # mock
        m = cfg["mock"]
        return DraftMeta(draft_id="mock", teams=int(m["teams"]), rounds=int(m["rounds"]), draft_type="snake",
                         reversal_round=0, pick_timer=int(m["pick_timer"]), my_slot=int(m["my_slot"]),
                         roster_positions=list(m["roster_positions"]))
    league, draft, my_slot, slot_names = resolve_draft(cfg, api)
    return DraftMeta.from_api(draft, my_slot, league["roster_positions"], slot_names)


# ---------------------------------------------------------------------------
def cmd_setup(cfg, args):
    api = Sleeper(cfg["data_dir"])
    league, draft, my_slot, slot_names = resolve_draft(cfg, api)
    meta = DraftMeta.from_api(draft, my_slot, league["roster_positions"], slot_names)
    api.players()
    slots = Slots.from_roster_positions(meta.roster_positions)
    info = {"league_id": league["league_id"], "name": league.get("name"), "season": league.get("season"),
            "draft_id": meta.draft_id,
            "status": draft.get("status"), "type": meta.draft_type, "teams": meta.teams, "rounds": meta.rounds,
            "pick_timer": meta.pick_timer, "reversal_round": meta.reversal_round, "my_slot": my_slot,
            "roster_positions": meta.roster_positions, "scoring_settings": league.get("scoring_settings"),
            "draft_order": {v: k for k, v in slot_names.items()}, "my_picks": my_pick_numbers(meta)}
    with open(os.path.join(cfg["data_dir"], "league.json"), "w") as f:
        json.dump(info, f, indent=1)
    print(f"League: {info['name']}  draft {meta.draft_id}  status={info['status']}  type={meta.draft_type}")
    print(f"Teams {meta.teams}  rounds {meta.rounds}  clock {meta.pick_timer}s  reversal_round {meta.reversal_round}")
    print(f"Roster: {' '.join(meta.roster_positions)}   (fixed {slots.fixed}, flex {len(slots.flex)}, bench {slots.bench}, ir {slots.ir}, ignored {slots.ignored})")
    print(f"My slot: {my_slot}   my picks: {info['my_picks']}")
    print("Draft order:", ", ".join(f"{s}:{n}" for s, n in sorted(slot_names.items())) or "(not posted yet)")
    sc = league.get("scoring_settings") or {}
    keys = ["pass_td", "pass_yd", "pass_int", "pass_2pt", "rush_td", "rush_yd", "rec", "rec_td", "rec_yd", "fum_lost",
            "bonus_rec_te", "bonus_rec_rb", "bonus_rec_wr", "rec_2pt", "rush_2pt"]
    print("\n--- paste into RESEARCH_PROMPT.md ---")
    print(f"Teams: {meta.teams}\nDraft: {meta.draft_type}, {meta.rounds} rounds\nRoster: {', '.join(meta.roster_positions)}")
    print("Scoring: " + ", ".join(f"{k}: {sc[k]}" for k in keys if k in sc))
    print(f"wrote {cfg['data_dir']}/league.json")
    print()
    do_fetch(cfg)   # seed the numeric board in the same step


def cmd_validate(cfg, args):
    api = None if args.offline else Sleeper(cfg["data_dir"])
    sp = api.players() if api else None
    res = load_research(cfg["data_dir"], sp)
    for p in res.problems:
        print(p)
    pl = res.players
    slots = Slots.from_roster_positions(_roster_positions(cfg, api))
    teams = _teams(cfg, api)
    repl = replacement_levels(pl, teams, slots, "proj", cfg["scoring"].get("replacement_extra", {}))
    print("\nReplacement level (proj):", {k: round(v, 1) for k, v in repl.items()})
    tt = tier_table(pl)
    print("\nTiers:")
    print(tt.to_string(index=False))
    out = os.path.join(cfg["data_dir"], "players_resolved.csv")
    pl.assign(vorp=(pl["proj"] - pl["pos"].map(repl)).round(1)).to_csv(out, index=False)
    print(f"\nwrote {out}")
    print("RESULT:", "OK" if res.ok else "ERRORS - fix before draft day")
    return 0 if res.ok else 1


def _roster_positions(cfg, api):
    lj = os.path.join(cfg["data_dir"], "league.json")
    if os.path.exists(lj):
        with open(lj) as f:
            return json.load(f)["roster_positions"]
    return cfg["mock"]["roster_positions"]


def _teams(cfg, api):
    lj = os.path.join(cfg["data_dir"], "league.json")
    if os.path.exists(lj):
        with open(lj) as f:
            return int(json.load(f)["teams"])
    return int(cfg["mock"]["teams"])


def cmd_script(cfg, args):
    """Pre-draft: expected availability at each of my picks from an empty board."""
    api = None if args.offline else Sleeper(cfg["data_dir"])
    sp = api.players() if api else None
    res = load_research(cfg["data_dir"], sp)
    if not res.ok:
        sys.exit("research has errors; run validate")
    meta = meta_from_cfg(cfg, api) if not args.offline else meta_from_cfg(cfg, None)
    if args.slot:
        meta.my_slot = int(args.slot)
    slots = Slots.from_roster_positions(meta.roster_positions)
    st = DraftState(meta)
    rng = np.random.default_rng(int(cfg.get("seed", 0)) or None)
    repl = replacement_levels(res.players, meta.teams, slots, "proj", cfg["scoring"].get("replacement_extra", {}))
    vorp = res.players["proj"] - res.players["pos"].map(repl)
    mc_cfg = dict(cfg["mc"])
    mc_cfg["sims"] = int(args.sims or mc_cfg["sims"])
    rows = []
    print(f"Slot {meta.my_slot}, picks {st.my_picks}. Survival from an empty board (ADP model, no autodrafters).")
    for pk in st.my_picks:
        upcoming = st.slots_between(1, pk)
        rnd, _ = round_and_pick(pk, meta.teams)
        r = simulate(res.players, set(), upcoming, st.rosters, set(), slots, rnd, meta.rounds, [], mc_cfg, rng, vorp=vorp)
        surv = pd.Series(r["surv"])
        top = res.players.loc[surv.index].assign(surv=surv.values, vorp=vorp.reindex(surv.index).values)
        likely = top[top["surv"] >= 0.5].sort_values("vorp", ascending=False)
        line = ", ".join(f"{n} {s*100:.0f}%" for n, s in zip(likely["name"].head(8), likely["surv"].head(8)))
        print(f"\nR{rnd} pick {pk}: E[best vorp] " + " ".join(f"{p}:{v:.0f}" for p, v in r["pos_expect"].items()))
        print(f"   likely available (>=50%): {line}")
        for pid, s in r["surv"].items():
            rows.append({"my_pick": pk, "round": rnd, "pid": pid, "name": res.players.at[pid, "name"],
                         "pos": res.players.at[pid, "pos"], "surv": round(s, 3)})
    os.makedirs(cfg["out_dir"], exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(cfg["out_dir"], "script.csv"), index=False)
    print(f"\nwrote {cfg['out_dir']}/script.csv")


def _run(cfg, client, meta, res, interval, headless=False, on_my_clock=None, max_ticks=None):
    dash = None
    if not headless:
        from .dashboard import Dashboard
        dash = Dashboard()
    eng = Engine(client, meta, res, cfg, dashboard=dash, on_my_clock=on_my_clock)
    eng.say(f"slot {meta.my_slot}, picks {eng.state.my_picks}")
    try:
        eng.run(interval=interval, max_ticks=max_ticks)
    except KeyboardInterrupt:
        pass
    finally:
        if dash:
            dash.stop()
    for m in eng.messages[-6:]:
        print(m)
    return eng


def cmd_run(cfg, args):
    api = Sleeper(cfg["data_dir"])
    res = load_research(cfg["data_dir"], api.players())
    if not res.ok:
        for p in res.problems:
            if p.startswith("ERROR"):
                print(p)
        sys.exit("research has errors; run validate")
    meta = meta_from_cfg(cfg, api)
    _run(cfg, api, meta, res, float(cfg["poll_interval"]), headless=args.headless)


def cmd_mock(cfg, args):
    from .mock import MockSleeper
    api = None if args.offline else Sleeper(cfg["data_dir"])
    sp = api.players() if api else None
    res = load_research(cfg["data_dir"], sp)
    if not res.ok:
        sys.exit("research has errors; run validate")
    meta = meta_from_cfg(cfg, None)
    if args.slot:
        meta.my_slot = int(args.slot)
    speed = float(args.speed or cfg["mock"]["speed"])
    # latencies are divided by speed in the mock, so the time-based detectors must be too
    cfg = dict(cfg)
    cfg["mc"] = dict(cfg["mc"], autodraft_latency_s=float(cfg["mc"]["autodraft_latency_s"]) / speed)
    cfg["llm"] = dict(cfg["llm"], eta_trigger_s=float(cfg["llm"].get("eta_trigger_s", 60)) / speed)
    client = MockSleeper(meta, res.players, cfg, speed=speed, seed=int(args.seed))
    if args.no_llm:
        cfg = dict(cfg)
        cfg["llm"] = dict(cfg["llm"], backend="none")

    def on_my_clock(eng):
        if eng.cands is not None and len(eng.cands):
            client.queue_my_pick(eng.cands.iloc[0]["pid"])

    print(f"mock draft: autodraft slots {sorted(client.autodraft)}, speed x{speed}")
    eng = _run(cfg, client, meta, res, max(0.2, float(cfg["poll_interval"]) / speed), headless=args.headless,
               on_my_clock=on_my_clock, max_ticks=args.max_ticks)
    print("\nFinal roster:")
    print(eng.my_roster()[["name", "pos", "team", "bye", "tier", "proj"]].to_string(index=False))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="sleeper-agent")
    ap.add_argument("-c", "--config", default="config.yaml")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("setup", help="fetch league/draft settings, cache players, print my picks")
    f = sub.add_parser("fetch", help="build players_skeleton.csv from Sleeper projections + schedule")
    f.add_argument("--season", type=int)
    rs = sub.add_parser("research", help="fan the judgment pass out across parallel agents and merge")
    rs.add_argument("--by", choices=["team", "division"], default="team",
                    help="shard grain: one agent per team (default, most accurate) or per division (faster)")
    rs.add_argument("--parallel", type=int, default=4)
    rs.add_argument("--model", default="")
    rs.add_argument("--timeout", type=float, default=None, help="override llm.research_timeout_s")
    rs.add_argument("--write-only", action="store_true", help="just write shard prompts for manual use in any LM")
    rs.add_argument("--from-replies", action="store_true", help="merge saved shard replies (shard_*_reply.md) into the final CSVs")
    rs.add_argument("--offline", action="store_true")
    v = sub.add_parser("validate", help="validate research CSVs and resolve Sleeper ids")
    v.add_argument("--offline", action="store_true", help="skip Sleeper; synthetic ids")
    s = sub.add_parser("script", help="pre-draft survival at each of my picks")
    s.add_argument("--slot", type=int)
    s.add_argument("--sims", type=int)
    s.add_argument("--offline", action="store_true")
    r = sub.add_parser("run", help="live draft assistant")
    r.add_argument("--headless", action="store_true", help="no dashboard; writes out/card_latest.md only")
    m = sub.add_parser("mock", help="dry run against a simulated draft")
    m.add_argument("--slot", type=int)
    m.add_argument("--speed", type=float)
    m.add_argument("--seed", type=int, default=1)
    m.add_argument("--offline", action="store_true")
    m.add_argument("--no-llm", action="store_true")
    m.add_argument("--headless", action="store_true")
    m.add_argument("--max-ticks", type=int)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    cmds = {"setup": cmd_setup, "fetch": cmd_fetch, "research": cmd_research, "validate": cmd_validate, "script": cmd_script,
            "run": cmd_run, "mock": cmd_mock}
    return cmds[args.cmd](cfg, args)


if __name__ == "__main__":
    sys.exit(main() or 0)
