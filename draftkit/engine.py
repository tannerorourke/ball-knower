"""The poll loop. Works against the real Sleeper client or the mock."""
from __future__ import annotations

import json
import os
import threading
import time

import numpy as np
import pandas as pd

from . import card as cardmod
from .data import Research, tier_table
from .montecarlo import simulate
from .scoring import Slots, candidate_table, roster_fill, replacement_levels, value_column
from .state import DraftMeta, DraftState, round_and_pick


class Engine:
    def __init__(self, client, meta: DraftMeta, research: Research, cfg: dict, dashboard=None, on_my_clock=None):
        self.client = client
        self.meta = meta
        self.res = research
        self.cfg = cfg
        self.dash = dashboard
        self.on_my_clock = on_my_clock            # callback(engine) used by mock mode to auto-pick
        self.slots = Slots.from_roster_positions(meta.roster_positions)
        mc_cfg = cfg.get("mc", {})
        self.state = DraftState(meta, autodraft_latency_s=float(mc_cfg.get("autodraft_latency_s", 4.0)),
                                forced_autodraft=mc_cfg.get("autodraft_slots") or [])
        self.rng = np.random.default_rng(int(cfg.get("seed", 0)) or None)
        self.out_dir = cfg.get("out_dir", "out")
        os.makedirs(self.out_dir, exist_ok=True)
        self.log_f = open(os.path.join(self.out_dir, "log.jsonl"), "a")
        self.mc: dict | None = None
        self.cands: pd.DataFrame | None = None
        self.det_card = ""
        self.llm_text = ""
        self.llm_status = ""
        self._llm_thread: threading.Thread | None = None
        self._llm_for_pick: int | None = None
        self._headsup_for_pick: int | None = None
        self.messages: list[str] = []
        self.players_pos = self.res.players["pos"].to_dict()

    # ------------------------------------------------------------------
    def log(self, kind: str, **kw):
        rec = {"t": time.time(), "kind": kind, **kw}
        self.log_f.write(json.dumps(rec, default=str) + "\n")
        self.log_f.flush()

    def say(self, msg: str):
        self.messages.append(f"{time.strftime('%H:%M:%S')} {msg}")
        self.messages = self.messages[-12:]
        self.log("msg", msg=msg)

    # ------------------------------------------------------------------
    def my_roster(self) -> pd.DataFrame:
        pids = [p for p in self.state.rosters.get(self.meta.my_slot, []) if p in self.res.players.index]
        return self.res.players.loc[pids]

    def unknown_picks(self) -> list[str]:
        """Taken player_ids not in the research table (drafted off-board)."""
        return [p for p in self.state.taken if p not in self.res.players.index]

    def opp_needs(self, until_pick: int | None) -> dict[int, str]:
        out = {}
        if until_pick is None:
            return out
        for slot in self.state.slots_between(self.state.current_pick_no, until_pick):
            pids = [p for p in self.state.rosters.get(slot, []) if p in self.res.players.index]
            roster = self.res.players.loc[pids]
            fo, fl, _ = roster_fill(roster, self.slots) if len(roster) else (dict(self.slots.fixed), len(self.slots.flex), [])
            need = [f"{p}x{n}" for p, n in fo.items() if n > 0]
            counts = roster["pos"].value_counts().to_dict() if len(roster) else {}
            have = " ".join(f"{p}{counts.get(p,0)}" for p in ["QB", "RB", "WR", "TE"])
            tag = "AUTO" if self.state.is_autodraft(slot) else "human"
            out[slot] = f"{tag} | has {have} | open starters: {' '.join(need) or 'none'} flex {fl}"
        return out

    # ------------------------------------------------------------------
    def recompute(self):
        st = self.state
        rnd, _ = round_and_pick(min(st.current_pick_no, self.meta.total_picks), self.meta.teams)
        horizon = st.horizon_pick()
        upcoming = st.slots_between(st.current_pick_no, horizon) if horizon else []
        # if I'm on the clock now, sim starts after my pick: same slots list (I'm excluded already)
        blend = {int(k): float(v) for k, v in self.cfg.get("scoring", {}).get("ceiling_blend", {1: 0.0}).items()}
        val = value_column(self.res.players, rnd, blend)
        df = self.res.players.assign(val=val)
        repl = replacement_levels(df, self.meta.teams, self.slots, "val", self.cfg.get("scoring", {}).get("replacement_extra", {}))
        vorp = df["val"] - df["pos"].map(repl)
        self.mc = simulate(self.res.players, st.taken, upcoming, st.rosters, st.autodraft_slots(), self.slots, rnd,
                           self.meta.rounds, st.recent_positions(self.players_pos), self.cfg.get("mc", {}), self.rng,
                           vorp=vorp)
        self.cands = candidate_table(self.res.players, st.taken, st.rosters.get(self.meta.my_slot, []), rnd,
                                     self.meta.rounds, self.slots, self.meta.teams, self.mc, self.res.stacks,
                                     self.res.anti, self.cfg.get("scoring", {}), top_n=int(self.cfg.get("top_n", 8)))
        self.tiers = tier_table(self.res.players, available=set(self.res.players.index) - st.taken)
        self.det_card = cardmod.deterministic_card(self.cands, self.tiers, st.picks_until_me())
        self.horizon = horizon

    # ------------------------------------------------------------------
    def _run_llm(self, pick_no: int, header: str):
        packet = cardmod.build_packet(self.state.summary(), self.cands, self.my_roster(), self.tiers,
                                      self.opp_needs(self.horizon), self.cfg.get("strategy_notes", ""), self.horizon)
        with open(os.path.join(self.out_dir, f"packet_{pick_no}.json"), "w") as f:
            json.dump(packet, f, indent=1, default=str)
        t0 = time.time()
        text, status = cardmod.llm_card(packet, self.cfg.get("llm", {}))
        if self._llm_for_pick != pick_no:
            return  # stale
        self.llm_text, self.llm_status = text, status
        self.say(f"LLM card ({status}) in {time.time()-t0:.0f}s")
        cardmod.write_card(self.out_dir, header, self.det_card, text or None, status)
        self.log("llm_card", pick_no=pick_no, status=status, text=text)

    def maybe_trigger_llm(self, header: str):
        st = self.state
        pick_no = st.current_pick_no
        want = st.on_clock_is_me or (self.cfg.get("llm", {}).get("at_n_minus_2") and st.picks_until_me() == 2)
        if not want or self._llm_for_pick == pick_no:
            return
        if self.cfg.get("llm", {}).get("backend", "none") == "none":
            return
        self._llm_for_pick = pick_no
        self.llm_text, self.llm_status = "", "running"
        self._llm_thread = threading.Thread(target=self._run_llm, args=(pick_no, header), daemon=True)
        self._llm_thread.start()

    # ------------------------------------------------------------------
    def tick(self, now: float | None = None) -> bool:
        """One poll. Returns True if the board changed."""
        now = now or time.time()
        try:
            picks = self.client.picks(self.meta.draft_id)
        except Exception as e:  # noqa
            self.say(f"poll error: {type(e).__name__}: {str(e)[:80]}")
            return False
        new = self.state.update(picks, now)
        if hasattr(self.client, "draft"):
            try:
                self.state.set_draft_info(self.client.draft(self.meta.draft_id))
            except Exception:
                pass
        for p in new:
            pid = str(p["player_id"])
            slot = int(p.get("draft_slot") or 0)
            nm = self.res.players["name"].get(pid) or (p.get("metadata") or {}).get("first_name", "") + " " + (p.get("metadata") or {}).get("last_name", "?")
            who = "ME" if slot == self.meta.my_slot else f"slot {slot}"
            self.say(f"pick {p['pick_no']}: {who} -> {nm}")
            self.log("pick", pick_no=p["pick_no"], slot=slot, pid=pid, name=nm)
        changed = bool(new) or self.cands is None
        if changed and not self.state.complete:
            self.recompute()
            s = self.state.summary(now)
            header = f"R{s['round']}.{s['pick_in_round']} | on clock: slot {s['on_clock_slot']}{' (ME)' if s['on_clock_is_me'] else ''} | {s['picks_until_me']} picks until me"
            cardmod.write_card(self.out_dir, header, self.det_card, None, "")
            with open(os.path.join(self.out_dir, "state.json"), "w") as f:
                json.dump({**s, "candidates": self.cands.to_dict("records")}, f, indent=1, default=str)
            if s["picks_until_me"] == 2 and self._headsup_for_pick != self.state.next_my_pick():
                self._headsup_for_pick = self.state.next_my_pick()
                top = ", ".join(f"{r['name']} ({r['surv']*100:.0f}%)" for _, r in self.cands.head(3).iterrows())
                self.say(f"HEADS UP: 2 picks out. Leading: {top}")
            if s["on_clock_is_me"]:
                self.log("my_turn", pick_no=self.state.current_pick_no, header=header,
                         candidates=self.cands.head(5).to_dict("records"), card=self.det_card)
            self.maybe_trigger_llm(header)
            if self.state.on_clock_is_me and self.on_my_clock:
                self.on_my_clock(self)
        return changed

    def run(self, interval: float = 2.0, max_ticks: int | None = None):
        n = 0
        while not self.state.complete:
            t0 = time.time()
            self.tick(t0)
            if self.dash:
                self.dash.render(self)
            n += 1
            if max_ticks and n >= max_ticks:
                break
            time.sleep(max(0.0, interval - (time.time() - t0)))
        if self.state.complete:
            self.say("Draft complete.")
            if self.dash:
                self.dash.render(self)
            self.write_final()

    def write_final(self):
        roster = self.my_roster()
        path = os.path.join(self.out_dir, "my_roster.csv")
        roster[["name", "pos", "team", "bye", "tier", "proj", "risk", "role", "tags"]].to_csv(path, index=False)
        self.say(f"wrote {path}")
