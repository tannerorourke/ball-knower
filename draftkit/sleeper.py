"""Thin client for Sleeper's public read-only API (no auth)."""
from __future__ import annotations

import json
import os
import time

import requests

BASE = "https://api.sleeper.app/v1"
POSITIONS = {"QB", "RB", "WR", "TE"}


class Sleeper:
    def __init__(self, data_dir: str = "data", timeout: float = 10.0):
        self.s = requests.Session()
        self.s.headers["User-Agent"] = "draftkit/0.1"
        self.data_dir = data_dir
        self.timeout = timeout
        os.makedirs(data_dir, exist_ok=True)

    def _get(self, path: str):
        r = self.s.get(f"{BASE}{path}", timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    # --- lookups -------------------------------------------------------
    def user(self, username: str) -> dict:
        return self._get(f"/user/{username}")

    def league(self, league_id: str) -> dict:
        return self._get(f"/league/{league_id}")

    def league_users(self, league_id: str) -> list:
        return self._get(f"/league/{league_id}/users")

    def league_rosters(self, league_id: str) -> list:
        return self._get(f"/league/{league_id}/rosters")

    def league_drafts(self, league_id: str) -> list:
        return self._get(f"/league/{league_id}/drafts")

    def draft(self, draft_id: str) -> dict:
        return self._get(f"/draft/{draft_id}")

    def picks(self, draft_id: str) -> list:
        return self._get(f"/draft/{draft_id}/picks")

    # --- undocumented but long-stable endpoints ------------------------
    def projections(self, season: int) -> list:
        """Full-season stat projections for all skill players, with Sleeper ADP."""
        url = ("https://api.sleeper.com/projections/nfl/{s}?season_type=regular"
               "&position[]=QB&position[]=RB&position[]=WR&position[]=TE&order_by=adp_half_ppr").format(s=season)
        r = self.s.get(url, timeout=30)
        r.raise_for_status()
        return r.json()

    def schedule(self, season: int) -> list:
        r = self.s.get(f"https://api.sleeper.app/schedule/nfl/regular/{season}", timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def byes(self, season: int) -> dict[str, int]:
        games = self.schedule(season)
        weeks = sorted({g["week"] for g in games})
        played: dict[str, set] = {}
        for g in games:
            for t in (g.get("home"), g.get("away")):
                if t:
                    played.setdefault(t, set()).add(g["week"])
        return {t: min(set(weeks) - wk) for t, wk in played.items() if set(weeks) - wk}

    # --- player table (5 MB; cache for a day) --------------------------
    def players(self, max_age_hours: float = 24.0) -> dict:
        path = os.path.join(self.data_dir, "players_nfl.json")
        if os.path.exists(path):
            age_h = (time.time() - os.path.getmtime(path)) / 3600
            if age_h < max_age_hours:
                with open(path) as f:
                    return json.load(f)
        data = self._get("/players/nfl")
        slim = {
            pid: {
                "player_id": pid,
                "name": p.get("full_name") or f"{p.get('first_name','')} {p.get('last_name','')}".strip(),
                "first_name": p.get("first_name"),
                "last_name": p.get("last_name"),
                "pos": p.get("position"),
                "team": p.get("team"),
                "status": p.get("status"),
                "injury_status": p.get("injury_status"),
                "search_rank": p.get("search_rank"),
                "age": p.get("age"),
                "years_exp": p.get("years_exp"),
            }
            for pid, p in data.items()
            if isinstance(p, dict) and p.get("position") in POSITIONS
        }
        with open(path, "w") as f:
            json.dump(slim, f)
        return slim
