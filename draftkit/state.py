"""Draft order math and live draft state."""
from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field


@dataclass
class DraftMeta:
    draft_id: str
    teams: int
    rounds: int
    draft_type: str                  # snake | linear
    reversal_round: int              # 0 = none; 3 = third-round reversal
    pick_timer: int                  # seconds
    my_slot: int
    roster_positions: list[str]
    slot_to_roster_id: dict[int, int] = field(default_factory=dict)
    slot_names: dict[int, str] = field(default_factory=dict)

    @classmethod
    def from_api(cls, draft: dict, my_slot: int, roster_positions: list[str], slot_names=None) -> "DraftMeta":
        st = draft.get("settings", {})
        return cls(
            draft_id=draft["draft_id"],
            teams=int(st.get("teams") or len(draft.get("slot_to_roster_id") or {}) or 12),
            rounds=int(st.get("rounds") or len(roster_positions)),
            draft_type=draft.get("type", "snake"),
            reversal_round=int(st.get("reversal_round") or 0),
            pick_timer=int(st.get("pick_timer") or 180),
            my_slot=my_slot,
            roster_positions=roster_positions,
            slot_to_roster_id={int(k): int(v) for k, v in (draft.get("slot_to_roster_id") or {}).items()},
            slot_names=slot_names or {},
        )

    @property
    def total_picks(self) -> int:
        return self.teams * self.rounds


def slot_for_pick(pick_no: int, teams: int, draft_type: str = "snake", reversal_round: int = 0) -> int:
    """1-based overall pick number -> 1-based draft slot."""
    r = (pick_no - 1) // teams + 1
    i = (pick_no - 1) % teams
    if draft_type == "linear":
        return i + 1
    forward = (r % 2 == 1)
    if reversal_round and r >= reversal_round:
        forward = not forward
    return i + 1 if forward else teams - i


def pick_order(meta: DraftMeta) -> list[int]:
    """Slot picking at each overall pick, index 0 = pick 1."""
    return [slot_for_pick(p, meta.teams, meta.draft_type, meta.reversal_round)
            for p in range(1, meta.total_picks + 1)]


def my_pick_numbers(meta: DraftMeta) -> list[int]:
    return [p for p, s in enumerate(pick_order(meta), start=1) if s == meta.my_slot]


def round_and_pick(pick_no: int, teams: int) -> tuple[int, int]:
    return (pick_no - 1) // teams + 1, (pick_no - 1) % teams + 1


class DraftState:
    """Mutable view of the draft built from successive /picks responses."""

    def __init__(self, meta: DraftMeta, autodraft_latency_s: float = 4.0, forced_autodraft: list[int] | None = None):
        self.meta = meta
        self.order = pick_order(meta)
        self.my_picks = my_pick_numbers(meta)
        self.picks: list[dict] = []                 # raw sleeper pick dicts, ordered by pick_no
        self.taken: set[str] = set()
        self.rosters: dict[int, list[str]] = {s: [] for s in range(1, meta.teams + 1)}
        self.latency: dict[int, list[float]] = {s: [] for s in range(1, meta.teams + 1)}
        self.last_pick_seen_at: float | None = None
        self.last_picked_ms: int | None = None      # from draft object, if polled
        self.autodraft_latency_s = autodraft_latency_s
        self.forced_autodraft = set(forced_autodraft or [])
        self.status = "pre_draft"
        self.started_at: float | None = None

    # --- ingestion -----------------------------------------------------
    def update(self, picks: list[dict], now: float | None = None) -> list[dict]:
        """Ingest a /picks payload. Returns the list of newly seen picks."""
        now = now or time.time()
        picks = sorted(picks, key=lambda p: p["pick_no"])
        new = picks[len(self.picks):]
        for i, p in enumerate(new):
            slot = int(p.get("draft_slot") or self.order[p["pick_no"] - 1])
            pid = str(p["player_id"])
            self.picks.append(p)
            self.taken.add(pid)
            self.rosters.setdefault(slot, []).append(pid)
            # latency: time since the previous pick was observed. Several picks in
            # one poll => all but the first were near-instant.
            if self.last_pick_seen_at is not None:
                lat = 0.0 if i > 0 else max(0.0, now - self.last_pick_seen_at)
                self.latency[slot].append(lat)
            self.last_pick_seen_at = now
        if new and self.started_at is None:
            self.started_at = now
        return new

    def set_draft_info(self, draft: dict):
        self.status = draft.get("status", self.status)
        lp = draft.get("last_picked")
        if lp:
            self.last_picked_ms = int(lp)

    # --- derived -------------------------------------------------------
    @property
    def current_pick_no(self) -> int:
        return len(self.picks) + 1

    @property
    def complete(self) -> bool:
        return len(self.picks) >= self.meta.total_picks or self.status == "complete"

    @property
    def on_clock_slot(self) -> int | None:
        if self.complete:
            return None
        return self.order[self.current_pick_no - 1]

    @property
    def on_clock_is_me(self) -> bool:
        return self.on_clock_slot == self.meta.my_slot

    def on_clock_since_s(self, now: float | None = None) -> float | None:
        now = now or time.time()
        if self.last_picked_ms:
            return max(0.0, now - self.last_picked_ms / 1000.0)
        if self.last_pick_seen_at is not None:
            return max(0.0, now - self.last_pick_seen_at)
        return None

    def next_my_pick(self, after: int | None = None) -> int | None:
        """My next pick strictly after `after` (default: current pick, inclusive)."""
        cur = self.current_pick_no if after is None else after + 1
        for p in self.my_picks:
            if p >= cur:
                return p
        return None

    def picks_until_me(self) -> int | None:
        nxt = self.next_my_pick()
        return None if nxt is None else nxt - self.current_pick_no

    def horizon_pick(self) -> int | None:
        """The pick the survival sim targets: my next pick that has >=1 opponent
        pick before it (skips the second pick of a consecutive pair)."""
        cur = self.current_pick_no
        cands = [p for p in self.my_picks if p >= cur]
        if not cands:
            return None
        start = cur
        for p in cands:
            if p - start >= 1 and any(self.order[q - 1] != self.meta.my_slot for q in range(start, p)):
                return p
            start = p + 1
        return None

    def slots_between(self, start_pick: int, end_pick: int) -> list[int]:
        """Slots picking at picks [start_pick, end_pick), excluding me."""
        return [self.order[p - 1] for p in range(start_pick, min(end_pick, self.meta.total_picks + 1))
                if self.order[p - 1] != self.meta.my_slot]

    def is_autodraft(self, slot: int) -> bool:
        if slot in self.forced_autodraft:
            return True
        lat = self.latency.get(slot, [])
        if len(lat) < 2:
            return False
        return statistics.median(lat) < self.autodraft_latency_s

    def autodraft_slots(self) -> set[int]:
        return {s for s in range(1, self.meta.teams + 1) if self.is_autodraft(s)}

    def median_latency(self, slot: int, default: float = 60.0) -> float:
        lat = self.latency.get(slot, [])
        return statistics.median(lat) if lat else default

    def eta_to_my_pick_s(self) -> float | None:
        nxt = self.next_my_pick()
        if nxt is None:
            return None
        return sum(self.median_latency(s) for s in self.slots_between(self.current_pick_no, nxt))

    def recent_positions(self, players_pos: dict[str, str], k: int = 4) -> list[str]:
        return [players_pos.get(str(p["player_id"]), "?") for p in self.picks[-k:]]

    def summary(self, now: float | None = None) -> dict:
        cur = self.current_pick_no
        r, pr = round_and_pick(min(cur, self.meta.total_picks), self.meta.teams)
        return {
            "polled_at": now or time.time(),
            "last_pick_no": len(self.picks),
            "round": r,
            "pick_in_round": pr,
            "on_clock_slot": self.on_clock_slot,
            "on_clock_is_me": self.on_clock_is_me,
            "picks_until_me": self.picks_until_me(),
            "eta_s": self.eta_to_my_pick_s(),
            "on_clock_since_s": self.on_clock_since_s(now),
            "autodraft_slots": sorted(self.autodraft_slots()),
            "complete": self.complete,
        }
