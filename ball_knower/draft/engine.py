"""Polling loop - against Sleeper client or mock."""
from __future__ import annotations

import json
import os
import queue
import threading
import time

import numpy as np
import pandas as pd

from . import card as cardmod
from ..core.data import POS, Research, tier_table
from .montecarlo import simulate, slot_tendencies
from ..core.scoring import Slots, candidate_table, roster_fill, roster_needs, replacement_levels, value_column
from .state import DraftMeta, DraftState, round_and_pick


class Engine:
    def __init__(self, client, meta: DraftMeta, research: Research, cfg: dict, dashboard=None, on_my_clock=None):
        self.client = client
        self.meta = meta
        self.res = research
        self.cfg = cfg
        self.llm_cfg = cfg.get("llm", {})
        self.dash = dashboard
        self.on_my_clock = on_my_clock            # callback(engine) used by mock mode to auto-pick
        self.slots = Slots.from_roster_positions(meta.roster_positions)
        mc_cfg = cfg.get("mc", {})
        self.poll_interval = float(cfg.get("poll_interval", 2.0))
        self.state = DraftState(meta, autodraft_latency_s=float(mc_cfg.get("autodraft_latency_s", 3.0)),
                                forced_autodraft=mc_cfg.get("autodraft_slots") or [],
                                batch_gap_s=self.poll_interval / 2)
        self.rng = np.random.default_rng(int(cfg.get("seed", 0)) or None)
        self.out_dir = cfg.get("out_dir", "out")
        os.makedirs(self.out_dir, exist_ok=True)
        self.log_f = open(os.path.join(self.out_dir, "log.jsonl"), "a")
        self.mc: dict | None = None
        self.cands: pd.DataFrame | None = None
        self._vorp: pd.Series | None = None
        self.det_card = ""
        # the Claude panel: text is NEVER cleared on trigger - it persists until a
        # finished job replaces it (draft-#1 bug: suggestions vanished right after a pick)
        self.llm_text = ""
        self.llm_status = ""
        self.llm_kind = ""
        self.llm_pick = None   # -- the pick the on-screen card was written for
        self.llm_started = None  # -- wall time the in-flight call began, None when idle
        self.llm_running = None  # -- job kind currently executing (not queued), None when idle
        self.llm_error = None    # -- job kind of the most recent failure; cleared by the next success
        self._llm_q: queue.Queue[tuple[str, int]] = queue.Queue()
        self._llm_seen: set[tuple[str, int]] = set()
        self._card_pick_done: set[int] = set()    # my picks with a precard/card job enqueued
        self._headsup_for_pick: int | None = None
        self.conn_state = "ok"                    # ok | retry10 | down
        self._retry_at: float | None = None
        self.messages: list[str] = []
        self.players_pos = self.res.players["pos"].to_dict()
        if self.llm_cfg.get("backend", "none") != "none":
            threading.Thread(target=self._llm_loop, daemon=True).start()

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
        """Taken player_ids not in the research table (drafted off-board, e.g. DEF)."""
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
            have = " ".join(f"{p}{counts.get(p, 0)}" for p in POS if self.slots.fixed.get(p) or p in ("QB", "RB", "WR", "TE"))
            tag = "AUTO" if self.state.is_autodraft(slot) else "human"
            style = ""
            t = getattr(self, "_tend", {}).get(slot)
            if t and tag == "human":
                favors = [p for p, v in (t.get("pos_shift") or {}).items() if v <= -0.08]
                style = f" | style: noise x{t['sig_mult']}" + (f", leans {'/'.join(favors)}" if favors else "")
            out[slot] = f"{tag} | has {have} | open starters: {' '.join(need) or 'none'} flex {fl}{style}"
        return out

    def best_available_by_pos(self, n: int = 2) -> dict[str, list[dict]]:
        """Top raw-vorp available per position - the LLM's sleeper escape hatch."""
        if self._vorp is None:
            return {}
        avail = self.res.players[~self.res.players.index.isin(self.state.taken)]
        surv = (self.mc or {}).get("surv", {})
        out = {}
        for p in POS:
            sub = avail[avail["pos"] == p]
            if not len(sub):
                continue
            top = sub.assign(vorp=self._vorp.reindex(sub.index)).nlargest(n, "vorp")
            out[p] = [{"name": r["name"], "bye": int(r["bye"]), "tier": int(r["tier"]),
                       "vorp": round(float(r["vorp"]), 1), "surv": round(float(surv.get(pid, 1.0)) * 100),
                       "risk": int(r["risk"])} for pid, r in top.iterrows()]
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
        self._vorp = vorp
        self._tend = slot_tendencies(st.picks, self.res.players, self.cfg.get("mc", {}))
        self.mc = simulate(self.res.players, st.taken, upcoming, st.rosters, st.autodraft_slots(), self.slots, rnd,
                           self.meta.rounds, st.recent_positions(self.players_pos), self.cfg.get("mc", {}), self.rng,
                           vorp=vorp, tendencies=self._tend)
        self.cands = candidate_table(self.res.players, st.taken, st.rosters.get(self.meta.my_slot, []), rnd,
                                     self.meta.rounds, self.slots, self.meta.teams, self.mc, self.res.stacks,
                                     self.res.anti, self.cfg.get("scoring", {}), top_n=int(self.cfg.get("top_n", 12)))
        self.tiers = tier_table(self.res.players, available=set(self.res.players.index) - st.taken)
        cl = {(r.pos, int(r.tier)): (0.0 if r.cliff != r.cliff else round(float(r.cliff), 1))
              for r in self.tiers.itertuples()}
        self.cands["cliff"] = [cl.get((r.pos, int(r.tier)), 0.0) for r in self.cands.itertuples()]
        self.det_card = cardmod.deterministic_card(self.cands, self.tiers, st.picks_until_me())
        self.horizon = horizon

    # ------------------------------------------------------------------
    # LLM worker: one FIFO queue, one thread. Jobs always run to completion -
    # nothing in flight is ever cancelled; ordering alone guarantees
    # post-pick analysis -> next card at the wheel.
    def _enqueue(self, kind: str, pick_no: int):
        if self.llm_cfg.get("backend", "none") == "none":
            return
        if kind in ("precard", "card"):
            if pick_no in self._card_pick_done:
                return
            self._card_pick_done.add(pick_no)
        key = (kind, pick_no)
        if key in self._llm_seen:
            return
        self._llm_seen.add(key)
        self._llm_q.put(key)

    def _stale(self, kind: str, pick_no: int) -> bool:
        """The one allowed drop: a job dequeued after the pick it targets is superseded."""
        if kind == "precard":
            return self.state.current_pick_no > pick_no
        if kind == "postpick":
            nxt = self.state.next_my_pick(after=pick_no)
            return nxt is not None and self.state.current_pick_no > nxt
        return False

    def _llm_loop(self):
        while True:
            kind, pick_no = self._llm_q.get()
            try:
                if self._stale(kind, pick_no):
                    self.log("llm_skip", job=kind, pick_no=pick_no)
                    continue
                self._run_job(kind, pick_no)
            except Exception as e:  # noqa
                # -- log() takes the record type positionally, so the job kind rides as 'job';
                #    passing it as 'kind' collides with that parameter. The handler is itself
                #    guarded: a raise here ends the thread and silently kills every later card.
                try:
                    self.llm_status = f"fallback:{type(e).__name__}"
                    self.llm_error = kind
                    self.log("llm_error", job=kind, pick_no=pick_no, err=str(e)[:200])
                except Exception:  # noqa
                    pass
            finally:
                # -- clearing the running marker is what flips the panel from running to finished
                self.llm_started = None
                self.llm_running = None

    def _header(self) -> str:
        s = self.state.summary()
        return (f"R{s['round']}.{s['pick_in_round']} | on clock: slot {s['on_clock_slot']}"
                f"{' (ME)' if s['on_clock_is_me'] else ''} | {s['picks_until_me']} picks until me")

    def _run_job(self, kind: str, pick_no: int):
        self.llm_status = f"{kind} running"
        self.llm_started = t0 = time.time()
        self.llm_running = kind
        if kind in ("precard", "card"):
            # packet built at dequeue time, not enqueue time - a queued job sees the current board
            packet = cardmod.build_packet(self.state.summary(), self.cands, self.my_roster(), self.tiers,
                                          self.opp_needs(self.horizon), self.cfg.get("strategy_notes", ""),
                                          self.horizon, self.best_available_by_pos())
            with open(os.path.join(self.out_dir, f"packet_{pick_no}.json"), "w") as f:
                json.dump(packet, f, indent=1, default=str)
            text, status = cardmod.llm_card(packet, self.llm_cfg)
            if text:
                self.llm_text, self.llm_kind, self.llm_pick = text, kind, pick_no
            self.llm_status = status
            self.llm_error = None if status == "llm" else kind
            self.say(f"LLM card for pick {pick_no} ({status}) in {time.time()-t0:.0f}s")
            cardmod.write_card(self.out_dir, self._header(), self.det_card, text or None, status)
            self.log("llm_card", pick_no=pick_no, status=status, text=text)
        else:  # postpick
            packet = self._postpick_packet(pick_no)
            text, status = cardmod.post_pick_card(packet, self.llm_cfg)
            if text:
                self.llm_text, self.llm_kind, self.llm_pick = text, "postpick", pick_no
            self.llm_status = status
            self.llm_error = None if status == "llm" else kind
            self.say(f"team analysis after pick {pick_no} ({status}) in {time.time()-t0:.0f}s")
            if text:
                with open(os.path.join(self.out_dir, f"teamstate_{pick_no}.md"), "w") as f:
                    f.write(text + "\n")
            self.log("llm_teamstate", pick_no=pick_no, status=status, text=text)

    def _postpick_packet(self, pick_no: int) -> dict:
        st = self.state
        roster = self.my_roster()
        ctx = roster_needs(roster, self.slots, self.cfg.get("scoring", {}))
        byes = roster["bye"].astype(int).value_counts().to_dict() if len(roster) else {}
        last = roster.iloc[-1] if len(roster) else None
        tl = self.tiers[self.tiers["left"] > 0] if getattr(self, "tiers", None) is not None else pd.DataFrame()
        return {
            "just_picked": {"name": last["name"], "pos": last["pos"], "bye": int(last["bye"])} if last is not None else None,
            "pick_no": pick_no,
            "my_roster": roster[["name", "pos", "team", "bye", "tier", "proj"]].to_dict("records") if len(roster) else [],
            "open_starters": {p: n for p, n in ctx["fixed_open"].items() if n} | ({"FLEX": ctx["flex_open"]} if ctx["flex_open"] else {}),
            "open_backups": {p: n for p, n in ctx["backup_open"].items() if n},
            "bye_distribution": byes,
            "my_remaining_picks": [q for q in st.my_picks if q >= st.current_pick_no],
            "rounds": self.meta.rounds,
            "tiers_remaining": tl.to_dict("records"),
            "strategy_notes": self.cfg.get("strategy_notes", ""),
        }

    def maybe_trigger_llm(self):
        st = self.state
        if st.on_clock_is_me:
            self._enqueue("card", st.current_pick_no)
            return
        llm = self.llm_cfg
        if not llm.get("at_n_minus"):
            return
        pum = st.picks_until_me()
        nxt = st.next_my_pick()
        if not pum or nxt is None:
            return
        rnd, _ = round_and_pick(min(st.current_pick_no, self.meta.total_picks), self.meta.teams)
        fire, why = pum <= int(llm.get("n_minus", 2)), f"{pum} picks out"
        if not fire and rnd >= 4:
            # time rule: a wall of autodrafters ahead means less time than the count suggests
            eta = st.eta_to_my_pick_s()
            if eta is not None and eta <= float(llm.get("eta_trigger_s", 60)):
                fire, why = True, f"~{eta:.0f}s to my pick ({pum} picks out)"
        if not fire:
            return
        self._enqueue("precard", nxt)
        if self._headsup_for_pick != nxt and self.cands is not None:
            self._headsup_for_pick = nxt
            top = ", ".join(f"{r['name']} ({r['surv']*100:.0f}%)" for _, r in self.cands.head(3).iterrows())
            self.say(f"HEADS UP: {why}. Leading: {top}")

    # ------------------------------------------------------------------
    def _poll_picks(self, now: float) -> list | None:
        """Fetch /picks with the retry ladder: immediate retry, then 10s, then a
        persistent red 'restart' state that still quietly retries every interval."""
        for attempt in (0, 1):
            try:
                picks = self.client.picks(self.meta.draft_id)
                if self.conn_state != "ok":
                    self.say("reconnected")
                self.conn_state, self._retry_at = "ok", None
                return picks
            except Exception as e:  # noqa
                err = e
        if self.conn_state == "ok":
            self.conn_state, self._retry_at = "retry10", now + 10.0
            self.say(f"poll error ({type(err).__name__}) - retrying in 10s")
        elif self.conn_state == "retry10":
            self.conn_state, self._retry_at = "down", now + self.poll_interval
            self.say("CONNECTION LOST - close and restart the program")
        else:
            self._retry_at = now + self.poll_interval
        self.log("conn", state=self.conn_state, err=f"{type(err).__name__}: {str(err)[:120]}")
        return None

    def tick(self, now: float | None = None) -> bool:
        """One poll. Returns True if the board changed."""
        now = now or time.time()
        if self._retry_at and now < self._retry_at:
            return False
        picks = self._poll_picks(now)
        if picks is None:
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
            if slot == self.meta.my_slot and self.llm_cfg.get("post_pick", True):
                # wheel guard: back-to-back picks get one analysis, after the second
                if self.state.next_my_pick() != p["pick_no"] + 1:
                    self._enqueue("postpick", p["pick_no"])
        changed = bool(new) or self.cands is None
        if changed and not self.state.complete:
            self.recompute()
            s = self.state.summary(now)
            cardmod.write_card(self.out_dir, self._header(), self.det_card, None, "")
            with open(os.path.join(self.out_dir, "state.json"), "w") as f:
                json.dump({**s, "candidates": self.cands.to_dict("records")}, f, indent=1, default=str)
            if s["on_clock_is_me"]:
                self.log("my_turn", pick_no=self.state.current_pick_no, header=self._header(),
                         candidates=self.cands.head(5).to_dict("records"), card=self.det_card)
            self.maybe_trigger_llm()
            if self.state.on_clock_is_me and self.on_my_clock:
                self.on_my_clock(self)
        return changed

    def run(self, interval: float = 2.0, max_ticks: int | None = None):
        self.poll_interval = interval
        self.state.batch_gap_s = interval / 2
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
