"""Fan-out research: shard the judgment pass by NFL team across parallel agents.

Two grains, both aligned to real cohesion: --by team (default, 32 agents; the
whole context is one team - maximum accuracy, timing is free pre-draft) and
--by division (8 agents of 4 teams; divisional weeks-15-17 rematches inform
po_sched and division previews let one search serve four teams). Never by
position: one team's news is one search context, and stacks/anti are intra-team
relations, so a shard that owns whole teams emits them without seeing the rest
of the league. Tiers are the one cross-board quantity, so tiers stay
deterministic (from fetch) no matter what shards say.

Modes:
  research --write-only      write shard prompts; run them in any LM yourself
  research                   also run them through the configured llm backend in parallel
  research --from-replies    merge replies you pasted into out/research/
Merge is defensive: locked columns are re-checked row by row against the
skeleton; a tampered row falls back to the skeleton row.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys

import requests
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from .data import PLAYER_COLS, load_research, norm_name

DIVISIONS = {
    "AFC East": ["BUF", "MIA", "NE", "NYJ"], "AFC North": ["BAL", "CIN", "CLE", "PIT"],
    "AFC South": ["HOU", "IND", "JAX", "TEN"], "AFC West": ["DEN", "KC", "LV", "LAC"],
    "NFC East": ["DAL", "NYG", "PHI", "WAS"], "NFC North": ["CHI", "DET", "GB", "MIN"],
    "NFC South": ["ATL", "CAR", "NO", "TB"], "NFC West": ["ARI", "LAR", "SF", "SEA"],
}
TEAM_DIV = {t: d for d, ts in DIVISIONS.items() for t in ts}

LOCKED = ["name", "pos", "team", "bye", "proj", "adp"]
JUDGE = ["floor", "ceil", "risk", "role", "inj", "po_sched", "tags"]

RULES = """Rules for the judgment columns (floor, ceil, risk, role, inj, po_sched, tags):
- role: LOCKED (unambiguous starter, >=65% of position's opportunity) | SHARED (committee/1B) | CONTESTED (job in question) | UNKNOWN.
- risk 0-5 int: 0 locked+healthy+prime age; 1 one minor concern; 2 moderate injury history OR mild role competition; 3 active injury designation OR real role competition OR age cliff (RB>=28, WR>=31, TE>=32); 4 two of those; 5 holdout/suspension/PUP/expected to miss >=3 games.
- inj: OK | Q | D | O | PUP | HOLDOUT | SUSP (correct the skeleton where news is fresher).
- floor ~20th-percentile season outcome, ceil ~80th; floor < proj < ceil always; widen for injury history, contested roles, rookies, new teams.
- po_sched: weeks 15-17 matchup quality for the position, 1 brutal .. 5 soft. Weeks 15-17 are heavy with divisional rematches - if your teams are a division, you know several of these defenses; use that.
- tags: 0-3 per row, |-separated, from exactly: NEW_OC, NEW_QB, NEW_TEAM, ROOKIE, ROOKIE_THREAT, HANDCUFF, RZ_HOG, YAC, DEEP, VOLUME, CONTRACT_YR, AGE_CLIFF, IR_STASH, INJ_HISTORY, BREAKOUT, DECLINE.
- kicker rows (pos K): role LOCKED unless an open camp battle; risk 0-2 in almost all cases; keep floor/ceil tight (kicking volume is offense-driven, not skill-variance); tags rarely apply.
stacks.csv: for each starting QB in YOUR teams, every RB/WR/TE on his team with >=12% projected target share (RB >=8%): qb,rec,team,stack_score,type,tgt_share,rz_share,adot. stack_score 0-100 (target share, red-zone share, QB pass volume; DEEP/high-aDOT profiles score the ceiling component higher than YAC). type: VOLUME|DEEP|RZ|YAC. Shares 0-1 two decimals, adot one decimal.
anti.csv: same-team cannibal pairs in YOUR teams: p1,p2,team,penalty,reason. Always include committee RB pairs (penalty 6-10), top two WRs of a low-volume passing offense (3-5), rushing-heavy RB with his own QB (1-2). reason: COMMITTEE|TARGET_SHARE|RUSH_QB|GOAL_LINE. Penalty in season points, one decimal."""

PROMPT = """You are one shard of a fantasy football research job. You own these NFL teams and no others:

  {teams}

Below is your slice of the draft board. The numeric columns name, pos, team, bye, proj, adp, tier were filled from projection data and are LOCKED: copy them through unchanged, do not add or drop rows, do not touch other teams' players.

League: {league}

Your job, for every row (players with adp <= 140 thoroughly; others only where you know something concrete):
1. If you have web access: news sweep, last 48 hours, for your teams - injuries, holdouts, suspensions, depth chart, coaching changes, trades. No web access: use what you know; prefer UNKNOWN/defaults over guessing.
2. Overwrite the judgment columns per the rules below.
3. Emit stacks.csv and anti.csv rows for YOUR teams only.

{rules}

OUTPUT: exactly three fenced code blocks, each opened with the filename as the info string, nothing else in your reply:
```players.csv
name,pos,team,bye,proj,floor,ceil,tier,adp,risk,role,inj,age,po_sched,tags
<every row from your slice, judgment columns updated>
```
```stacks.csv
qb,rec,team,stack_score,type,tgt_share,rz_share,adot
<your teams' rows>
```
```anti.csv
p1,p2,team,penalty,reason
<your teams' rows>
```

YOUR SLICE:
```csv
{slice_csv}
```"""

BLOCK_RE = re.compile(r"```(players\.csv|stacks\.csv|anti\.csv)\s*\n(.*?)```", re.S)


def league_line(data_dir: str) -> str:
    lj = os.path.join(data_dir, "league.json")
    if not os.path.exists(lj):
        return "12-team half-PPR snake (details unavailable - run `python -m sleeper_agent setup`)"
    with open(lj) as f:
        info = json.load(f)
    sc = info.get("scoring_settings") or {}
    keys = ["pass_td", "pass_yd", "pass_int", "rush_td", "rush_yd", "rec", "rec_td", "rec_yd", "fum_lost"]
    return (f"{info.get('teams')} teams, {info.get('type')} draft, {info.get('rounds')} rounds, "
            f"roster {' '.join(info.get('roster_positions', []))}, scoring "
            + ", ".join(f"{k}:{sc[k]}" for k in keys if k in sc))


def make_shards(skeleton: pd.DataFrame, by: str = "team") -> list[tuple[str, list[str]]]:
    """Shards as (label, teams). by='team': one agent per NFL team. by='division':
    one agent per division; team codes outside the division map (relocations)
    join the smallest division shard."""
    present = set(skeleton["team"].unique())
    if by == "team":
        return [(t if t not in TEAM_DIV else f"{t} ({TEAM_DIV[t]})", [t]) for t in sorted(present)]
    if by != "division":
        raise ValueError(f"unknown shard mode {by!r}: use 'team' or 'division'")
    shards = [(d, [t for t in ts if t in present]) for d, ts in DIVISIONS.items()]
    for t in sorted(present - set(TEAM_DIV)):
        shards.sort(key=lambda x: len(x[1]))
        shards[0][1].append(t)
    return [(d, sorted(ts)) for d, ts in shards if ts]


def _slug(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", label.split("(")[0]).strip("_")


def write_shards(data_dir: str, out_dir: str, by: str) -> list[str]:
    skel_path = os.path.join(data_dir, "players_skeleton.csv")
    if not os.path.exists(skel_path):
        sys.exit("no players_skeleton.csv - run `python -m sleeper_agent fetch` first")
    skeleton = pd.read_csv(skel_path)
    rdir = os.path.join(out_dir, "research")
    os.makedirs(rdir, exist_ok=True)
    for f in os.listdir(rdir):                       # stale prompts from a previous grain
        if re.fullmatch(r"shard_[A-Za-z0-9_]+_prompt\.md", f):
            os.remove(os.path.join(rdir, f))
    lg = league_line(data_dir)
    paths = []
    for i, (label, teams) in enumerate(make_shards(skeleton, by), 1):
        sl = skeleton[skeleton["team"].isin(teams)]
        title = f"the {label}: {', '.join(teams)}" if label in DIVISIONS else label
        prompt = PROMPT.format(teams=title, league=lg, rules=RULES,
                               slice_csv=sl.to_csv(index=False).strip())
        p = os.path.join(rdir, f"shard_{i:02d}_{_slug(label)}_prompt.md")
        with open(p, "w") as f:
            f.write(prompt)
        paths.append(p)
    return paths


def run_claude(prompt_path: str, model: str, timeout_s: float, tools: str, effort: str = "high") -> str:
    exe = shutil.which("claude")
    if not exe:
        raise RuntimeError("claude CLI not found on PATH")
    args = [exe, "-p", "--output-format", "text", "--allowedTools", tools, "--effort", effort]
    if model:
        args += ["--model", model]
    with open(prompt_path) as f:
        prompt = f.read()
    res = subprocess.run(args, input=prompt, capture_output=True, text=True, timeout=timeout_s)
    if res.returncode != 0:
        raise RuntimeError(f"claude exit {res.returncode}: {res.stderr[:200]}")
    return res.stdout


def run_openai_compat(prompt: str, llm_cfg: dict, timeout_s: float) -> str:
    """Any provider speaking the OpenAI chat-completions schema: OpenRouter,
    OpenAI, Gemini's compat endpoint, Groq, local Ollama/vLLM. No SDK."""
    base = (llm_cfg.get("base_url") or "").rstrip("/")
    if not base:
        raise RuntimeError("llm.base_url not set (e.g. https://openrouter.ai/api/v1)")
    key = os.environ.get(llm_cfg.get("api_key_env", "LLM_API_KEY"), "")
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    r = requests.post(f"{base}/chat/completions", headers=headers, timeout=timeout_s,
                      json={"model": llm_cfg.get("model", ""),
                            "messages": [{"role": "user", "content": prompt}]})
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def run_shard(prompt_path: str, llm_cfg: dict, model: str, timeout_s: float) -> str:
    backend = llm_cfg.get("backend", "claude-cli")
    if backend == "openai-compat":
        with open(prompt_path) as f:
            cfg = dict(llm_cfg, model=model)
            return run_openai_compat(f.read(), cfg, timeout_s)
    return run_claude(prompt_path, model, timeout_s, "WebSearch,WebFetch",
                      llm_cfg.get("research_effort") or "high")


def parse_reply(text: str) -> dict[str, pd.DataFrame]:
    import io
    out = {}
    for name, body in BLOCK_RE.findall(text):
        try:
            out[name] = pd.read_csv(io.StringIO(body.strip()), dtype=str, keep_default_na=False)
        except Exception:  # noqa
            pass
    return out


def merge(data_dir: str, out_dir: str, replies: list[dict[str, pd.DataFrame]]) -> list[str]:
    """Assemble data/players.csv, stacks.csv, anti.csv from shard replies.
    Locked columns are restored from the skeleton row-by-row; missing or
    tampered rows fall back to the skeleton entirely."""
    notes = []
    skeleton = pd.read_csv(os.path.join(data_dir, "players_skeleton.csv"))
    skeleton["_key"] = skeleton["name"].str.strip()
    judged = {}
    stacks, anti = [], []
    for i, rep in enumerate(replies, 1):
        pl = rep.get("players.csv")
        if pl is None or "name" not in getattr(pl, "columns", []):
            notes.append(f"shard {i}: no parsable players.csv block - its teams keep skeleton defaults")
            continue
        for _, r in pl.iterrows():
            judged[str(r["name"]).strip()] = r
        if "stacks.csv" in rep and len(rep["stacks.csv"]):
            stacks.append(rep["stacks.csv"])
        if "anti.csv" in rep and len(rep["anti.csv"]):
            anti.append(rep["anti.csv"])

    rows, fell_back = [], 0
    for _, sk in skeleton.iterrows():
        r = judged.get(sk["_key"])
        row = sk[PLAYER_COLS].to_dict()
        if r is None:
            fell_back += 1
        else:
            for c in JUDGE:
                if c in r and str(r[c]).strip() != "":
                    row[c] = str(r[c]).strip()
            try:  # floor < proj < ceil or revert to skeleton defaults
                if not (float(row["floor"]) < float(row["proj"]) < float(row["ceil"])):
                    row["floor"], row["ceil"] = sk["floor"], sk["ceil"]
            except ValueError:
                row["floor"], row["ceil"] = sk["floor"], sk["ceil"]
        rows.append(row)
    if fell_back:
        notes.append(f"{fell_back} rows had no shard judgment; kept skeleton defaults")

    pd.DataFrame(rows)[PLAYER_COLS].to_csv(os.path.join(data_dir, "players.csv"), index=False)

    # canonicalize stacks/anti names to the board: fix punctuation drift
    # ("Brian Thomas Jr" -> "Brian Thomas Jr."), drop rows naming off-board players
    canon = {norm_name(n): n for n in skeleton["name"]}
    for name, frames, cols, name_cols in [
            ("stacks.csv", stacks, ["qb", "rec", "team", "stack_score", "type", "tgt_share", "rz_share", "adot"], ["qb", "rec"]),
            ("anti.csv", anti, ["p1", "p2", "team", "penalty", "reason"], ["p1", "p2"])]:
        if frames:
            df = pd.concat(frames, ignore_index=True)
            df = df[[c for c in cols if c in df.columns]].drop_duplicates()
            dropped = []
            for c in name_cols:
                df[c] = df[c].map(lambda x: canon.get(norm_name(x), x))
                bad = ~df[c].isin(set(skeleton["name"]))
                dropped += df.loc[bad, c].tolist()
                df = df[~bad]
            if dropped:
                notes.append(f"{name}: dropped rows naming off-board players: {sorted(set(dropped))}")
            df.to_csv(os.path.join(data_dir, name), index=False)
        else:
            notes.append(f"no {name} rows from any shard")
    return notes


def cmd_research(cfg, args):
    out_dir = cfg["out_dir"]
    rdir = os.path.join(out_dir, "research")
    if args.from_replies:
        paths = sorted(f for f in os.listdir(rdir) if re.fullmatch(r"shard_[A-Za-z0-9_]+_reply\.(md|txt)", f))
        # only replies matching the current prompt set; stale ones from another grain are skipped
        current = {f[:-len("_prompt.md")] for f in os.listdir(rdir) if f.endswith("_prompt.md")}
        skipped = [p for p in paths if re.sub(r"_reply\.(md|txt)$", "", p) not in current]
        paths = [p for p in paths if p not in skipped]
        if skipped:
            print(f"skipping {len(skipped)} reply file(s) not matching the current shard set: {skipped}")
        if not paths:
            sys.exit(f"no shard replies matching the current prompts in {rdir}")
        replies = [parse_reply(open(os.path.join(rdir, p)).read()) for p in paths]
    else:
        if not args.offline:
            from .fetch import do_fetch
            try:
                do_fetch(cfg)          # fresh ADP/injury flags at research time; ~3s
            except Exception as e:  # noqa
                print(f"NOTE: skeleton refresh failed ({type(e).__name__}); using the existing one")
        prompts = write_shards(cfg["data_dir"], out_dir, args.by)
        print(f"wrote {len(prompts)} shard prompts to {rdir}/")
        if args.write_only:
            print("run each in any LM; save each reply beside its prompt with _prompt -> _reply in the "
                  "name (e.g. shard_01_ARI_reply.md); then: research --from-replies")
            return
        llm = cfg.get("llm", {})
        model = args.model or llm.get("research_model") or ""
        timeout = float(args.timeout) if args.timeout is not None else float(llm.get("research_timeout_s") or 900)
        backend = llm.get("backend", "claude-cli")
        if backend == "none":
            sys.exit("llm.backend is 'none' - research needs an LLM; set llm.backend or use --write-only")
        print(f"running {len(prompts)} shards via {backend} / {model or 'backend default'} (parallel {args.parallel}) ...")

        def one(p):
            try:
                text = run_shard(p, dict(llm, backend=backend), model, timeout)
                with open(p.replace("_prompt.md", "_reply.md"), "w") as f:
                    f.write(text)
                print(f"  done {os.path.basename(p)}")
                return parse_reply(text)
            except Exception as e:  # noqa
                print(f"  FAIL {os.path.basename(p)}: {e}")
                return {}

        with ThreadPoolExecutor(max_workers=int(args.parallel)) as ex:
            replies = list(ex.map(one, prompts))

    for note in merge(cfg["data_dir"], out_dir, replies):
        print("NOTE:", note)
    print("\nmerged -> data/players.csv, stacks.csv, anti.csv; validating:")
    res = load_research(cfg["data_dir"], None if args.offline else __import__("sleeper_agent.sleeper", fromlist=["Sleeper"]).Sleeper(cfg["data_dir"]).players())
    for p in res.problems:
        print(p)
    print("RESULT:", "OK" if res.ok else "ERRORS - fix before draft day")
    return 0 if res.ok else 1
