"""A fake Sleeper draft for dry runs. Same interface as the real client (picks/draft)."""
from __future__ import annotations

import time

import numpy as np

from .montecarlo import sample_one_pick
from .scoring import Slots
from .state import DraftMeta, pick_order, round_and_pick


class MockSleeper:
    def __init__(self, meta: DraftMeta, players, cfg: dict, speed: float = 10.0, seed: int = 1):
        self.meta = meta
        self.players = players
        self.cfg = cfg
        self.speed = speed                      # wall-clock divisor: 10 => a 60s pick takes 6s
        self.rng = np.random.default_rng(seed)
        self.order = pick_order(meta)
        self.slots = Slots.from_roster_positions(meta.roster_positions)
        mcfg = cfg.get("mock", {})
        n_auto = int(mcfg.get("autodraft_teams", 3))
        others = [s for s in range(1, meta.teams + 1) if s != meta.my_slot]
        self.autodraft = set(self.rng.choice(others, size=min(n_auto, len(others)), replace=False).tolist())
        self.human_lat = {s: float(self.rng.uniform(mcfg.get("human_min_s", 8), mcfg.get("human_max_s", 45))) for s in others}
        self.my_delay = float(mcfg.get("my_pick_delay_s", 20))
        self._picks: list[dict] = []
        self._taken: set[str] = set()
        self._rosters = {s: [] for s in range(1, meta.teams + 1)}
        self._next_at = time.time() + 1.0
        self._my_choice: str | None = None
        self._my_choice_at: float | None = None
        self.last_picked = int(time.time() * 1000)

    # -- the harness registers what it would pick ---------------------------
    def queue_my_pick(self, pid: str, delay_s: float | None = None):
        self._my_choice = pid
        self._my_choice_at = time.time() + (self.my_delay if delay_s is None else delay_s) / self.speed

    def _make_pick(self, slot: int, pid: str):
        pick_no = len(self._picks) + 1
        r, _ = round_and_pick(pick_no, self.meta.teams)
        p = self.players.loc[pid]
        self._picks.append({"pick_no": pick_no, "round": r, "draft_slot": slot, "roster_id": slot, "player_id": pid,
                            "metadata": {"first_name": p["name"].split()[0], "last_name": " ".join(p["name"].split()[1:]),
                                         "position": p["pos"], "team": p["team"]}})
        self._taken.add(pid)
        self._rosters[slot].append(pid)
        self.last_picked = int(time.time() * 1000)

    def _advance(self):
        now = time.time()
        while len(self._picks) < self.meta.total_picks:
            pick_no = len(self._picks) + 1
            slot = self.order[pick_no - 1]
            r, _ = round_and_pick(pick_no, self.meta.teams)
            if slot == self.meta.my_slot:
                if self._my_choice and self._my_choice_at and now >= self._my_choice_at and self._my_choice not in self._taken:
                    self._make_pick(slot, self._my_choice)
                    self._my_choice = None
                    self._next_at = now
                    continue
                # timeout -> autopick best adp available (what Sleeper does with no queue)
                if now - self.last_picked / 1000 > self.meta.pick_timer / self.speed:
                    avail = self.players[~self.players.index.isin(self._taken)].sort_values("adp")
                    self._make_pick(slot, avail.index[0])
                    self._my_choice = None
                    continue
                return
            lat = 0.5 if slot in self.autodraft else self.human_lat[slot] * float(self.rng.uniform(0.4, 1.6))
            due = self.last_picked / 1000 + lat / self.speed
            if now < due:
                return
            recent = [self.players["pos"].get(str(p["player_id"]), "?") for p in self._picks[-4:]]
            pid = sample_one_pick(self.players, self._taken, slot, self._rosters, self.autodraft, self.slots, r,
                                  self.meta.rounds, recent, self.cfg.get("mc", {}), self.rng)
            self._make_pick(slot, pid)

    # -- Sleeper-shaped API ------------------------------------------------
    def picks(self, draft_id: str) -> list[dict]:
        self._advance()
        return list(self._picks)

    def draft(self, draft_id: str) -> dict:
        status = "complete" if len(self._picks) >= self.meta.total_picks else "drafting"
        return {"draft_id": draft_id, "status": status, "last_picked": self.last_picked,
                "settings": {"teams": self.meta.teams, "rounds": self.meta.rounds, "pick_timer": self.meta.pick_timer}}
