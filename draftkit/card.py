"""The decision card: deterministic top-5 always; LLM-adjudicated version when available.
Also the post-pick team-state analysis."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time

import pandas as pd

from .data import norm_name

CARD_FORMAT = """A  <name>  <pos> T<tier>  vorp <n>  surv <n>%  bye <n>  stack <+n|->  need <tag>  via <CLIFF|SURV|NEED|BACKUP|STACK|VALUE|SLEEPER>
   + <one line, the strongest reason>
   - <one line, the strongest objection>
   ! <only when warranted: would be 3rd+ player on that bye week, or risk >= 3 - say why>
B  ...  (same shape; the ! line only when warranted)
C  ...
D  ...
E  ...
IF A gone -> <letter>: <one clause>
DATA <name>: <one clause - echo the packet's data_pick verbatim; why the data likes it>
SLEEPER <name>: <one clause - your considered pick; name the criterion or strategy note driving it>"""

SYSTEM = """You are adjudicating a fantasy football draft pick. You receive a packet of precomputed numbers for a closed set of candidates. You do not compute, estimate, or recall anything about players beyond the packet - every number you print must be copied from it. You present five acceptable candidates as PEERS, not a ranking: each passes a different criterion (a tier cliff, low survival, an open slot, a backup need, a stack, raw value). Explain each in one + line and one - line, plus a ! warning line only where warranted. You close with two picks: DATA (echo the packet's data_pick - the harness's answer) and SLEEPER (your own considered call, which may equal DATA; if it deviates, cite the criterion or strategy note that drives it). Your output is displayed to a human who has under three minutes to click. Be terse. No preamble, no closing remarks, no markdown headers."""

POSTPICK_SYSTEM = """You are reviewing a fantasy football roster mid-draft, immediately after the drafter made a pick. You receive the roster, its open starter and backup slots, the bye-week distribution, the remaining pick numbers, tiers remaining, and the drafter's strategy notes. You do not recall anything about players beyond the packet. Output at most 8 short lines: line 1 'ON PLAN' or 'OFF PLAN: <one clause>' vs the strategy notes; then the top 1-2 roster holes to target next (with the pick numbers to do it); then any bye pile-up forming (3+ players one week); then one line on the just-made pick. Be terse. No preamble, no markdown."""

VIA_TAGS = ("CLIFF", "SURV", "NEED", "BACKUP", "STACK", "VALUE", "SLEEPER")


def _via(r: pd.Series) -> str:
    if r["need"] == "STARTER":
        return "NEED"
    if r["need"] == "BACKUP":
        return "BACKUP"
    if r["tier_empty_p"] >= 0.5:
        return "CLIFF"
    if r["surv"] < 0.4:
        return "SURV"
    if r["stack"] > 0:
        return "STACK"
    return "VALUE"


def _warnings(r: pd.Series) -> list[str]:
    out = []
    if r["bye_team_count"] >= 2:
        out.append(f"bye {r['bye']}: would be player #{int(r['bye_team_count']) + 1} on this bye ({r['bye_with'] or 'team-wide'})")
    if r["risk"] >= 3:
        why = f"role {r['role']}, inj {r['inj']}" + (f", {r['tags']}" if r.get("tags") else "")
        out.append(f"risk {r['risk']}: {why}")
    return out


def data_pick(cands: pd.DataFrame) -> pd.Series:
    """The quick answer: highest adj that doesn't pile a 3rd player onto one bye."""
    ok = cands[cands["bye_team_count"] < 2]
    return (ok if len(ok) else cands).iloc[0]


def _fmt_line(letter: str, r: pd.Series) -> str:
    stack = f"stack +{r['stack']:.0f}" if r["stack"] > 0 else "stack -"
    return (f"{letter}  {r['name']:<22} {r['pos']} T{r['tier']}  vorp {r['vorp']:>5.1f}  surv {r['surv']*100:>3.0f}%  "
            f"bye {r['bye']}  {stack}  need {r['need']}  via {_via(r)}")


def deterministic_card(cands: pd.DataFrame, tiers: pd.DataFrame, picks_until_next: int | None) -> str:
    if cands.empty:
        return "(no candidates)"
    lines = []
    top = cands.head(5).reset_index(drop=True)
    for i, r in top.iterrows():
        letter = "ABCDE"[i]
        lines.append(_fmt_line(letter, r))
        plus, minus = [], []
        tl = tiers[(tiers.pos == r["pos"]) & (tiers.tier == r["tier"])]
        left = int(tl["left"].iloc[0]) if len(tl) else 0
        cliff = float(tl["cliff"].iloc[0]) if len(tl) and pd.notna(tl["cliff"].iloc[0]) else 0.0
        if r["wait_cost"] > 0:
            plus.append(f"waiting costs ~{r['wait_cost']:.0f} pts at {r['pos']} (tier closes p={r['tier_empty_p']*100:.0f}%, {left} left, cliff {cliff:.0f})")
        if r["stack"] > 0:
            plus.append(f"stack with {r['stack_with']}")
        if r["need"] == "STARTER":
            plus.append("fills an open starting slot")
        if r["need"] == "BACKUP":
            plus.append("fills a direct backup slot" + (f" ({r['need_note']})" if r["need_note"] else ""))
        if r["need"] == "DEPTH" and r["need_note"]:
            minus.append(f"depth pick while {r['need_note']}")
        if r["anti"] > 0:
            minus.append(f"cannibalizes {r['anti_with']} (-{r['anti']:.0f})")
        if r["risk"] >= 3:
            minus.append(f"risk {r['risk']}, role {r['role']}, inj {r['inj']}")
        if r["surv"] >= 0.6 and r["wait_cost"] <= 2:
            minus.append(f"likely still here next turn ({r['surv']*100:.0f}%) - can wait")
        lines.append("   + " + (plus[0] if plus else f"best adjusted value ({r['adj']:.1f})"))
        lines.append("   - " + (minus[0] if minus else f"risk {r['risk']}, {r['role']}"))
        for w in _warnings(r):
            lines.append("   ! " + w)
    if len(top) >= 2:
        lines.append(f"IF A gone -> B ({top.loc[1, 'name']})")
    dp = data_pick(cands)
    lines.append(f"DATA {dp['name']}: top adj ({dp['adj']:.1f}) without a 3rd-player bye stack")
    return "\n".join(lines)


def build_packet(summary: dict, cands: pd.DataFrame, roster: pd.DataFrame, tiers: pd.DataFrame,
                 opp_needs: dict[int, str], strategy_notes: str, horizon_pick: int | None,
                 best_by_pos: dict[str, list[dict]] | None = None) -> dict:
    keep = ["name", "pos", "team", "bye", "tier", "proj", "vorp", "adp", "surv", "wait_cost", "need", "need_pts",
            "need_note", "bye_with", "bye_team_count", "stack", "stack_with", "anti", "anti_with",
            "risk", "role", "inj", "tags", "tier_empty_p", "cliff", "adj"]
    c = cands[[k for k in keep if k in cands.columns]].copy()
    c["surv"] = (c["surv"] * 100).round(0)
    c["tier_empty_p"] = (c["tier_empty_p"] * 100).round(0)
    tl = tiers[tiers["left"] > 0].copy()
    tl["cliff"] = tl["cliff"].round(1)
    dp = data_pick(cands)
    return {
        "draft": {k: summary.get(k) for k in ["round", "pick_in_round", "picks_until_me", "on_clock_is_me"]},
        "my_next_pick_after_this": horizon_pick,
        "my_roster": roster[["name", "pos", "team", "bye", "tier", "proj"]].to_dict("records") if len(roster) else [],
        "candidates_ranked_by_adj": c.to_dict("records"),
        "data_pick": {"name": dp["name"], "pos": dp["pos"], "adj": float(dp["adj"])},
        "best_available_by_pos": best_by_pos or {},
        "tiers_remaining": tl.to_dict("records"),
        "opponent_needs_before_my_next_pick": opp_needs,
        "strategy_notes": strategy_notes,
        "column_glossary": {
            "vorp": "season pts over positional replacement (league-specific)",
            "surv": "% of sims in which this player is still available at my next pick",
            "wait_cost": "vorp minus expected best vorp at this position at my next pick; what waiting costs",
            "need": "STARTER=open fixed slot, FLEX=open flex slot, BACKUP=direct backup of my starters, "
                    "DEPTH=past starters+backups (gated), CAPPED=too many at pos, K-WAIT=kicker before the endgame",
            "need_note": "why the need tag was shaded (bye overlap, open slots elsewhere)",
            "bye_team_count": "how many players I already roster on this bye week (2+ = warn: pick would be the 3rd+)",
            "stack": "pts bonus for correlation with my QB/receivers",
            "anti": "pts penalty for cannibalizing a player I roster",
            "risk": "0-5 research judgment; 3+ = active injury/role/age concern (see role, inj, tags)",
            "tier_empty_p": "% of sims in which this player's tier is gone by my next pick",
            "cliff": "proj pts dropoff from this tier's floor to the next tier's best",
            "adj": "deterministic combined score; the harness's own ranking (byes are NOT scored, only warned)",
            "data_pick": "the harness's quick answer: highest adj avoiding a 3rd-player bye stack - echo it in the DATA line",
            "best_available_by_pos": "top raw-vorp available per position even if outside the ranked list; SLEEPER may cite one only with a strategy_notes or cliff justification",
        },
    }


def _validate(text: str, packet: dict) -> tuple[bool, str]:
    cands = pd.DataFrame(packet["candidates_ranked_by_adj"])
    names = {norm_name(n) for n in cands["name"]}
    extra = {norm_name(r["name"]) for rows in (packet.get("best_available_by_pos") or {}).values() for r in rows}
    found = 0
    data_line = sleeper_line = None
    for line in text.splitlines():
        s = line.strip()
        if len(s) > 2 and s[0] in "ABCDE" and s[1] == " ":
            body = norm_name(s[1:])
            if any(n and n in body for n in names):
                found += 1
            else:
                return False, f"option line names a player outside the candidate set: {s[:60]}"
        elif s.startswith("DATA"):
            data_line = s
        elif s.startswith("SLEEPER"):
            sleeper_line = s
    if found < 3:
        return False, "fewer than three option lines parsed"
    if data_line is None or norm_name(packet["data_pick"]["name"]) not in norm_name(data_line):
        return False, "DATA line missing or does not echo the packet's data_pick"
    if sleeper_line is None or not any(n and n in norm_name(sleeper_line) for n in names | extra):
        return False, "SLEEPER line missing or names a player outside the packet"
    return True, ""


def _resolve(cfg: dict) -> tuple[str, str, float]:
    """(backend, model, timeout) for the draft phase."""
    backend = (cfg or {}).get("backend", "none")
    return backend, cfg.get("draft_model") or "", float(cfg.get("draft_timeout_s") or 90)


def _call_backend(system: str, prompt: str, cfg: dict) -> tuple[str, str]:
    """Returns (text, error). error empty on success."""
    backend, model, timeout = _resolve(cfg)
    try:
        if backend == "anthropic":
            import anthropic  # noqa
            client = anthropic.Anthropic()
            msg = client.messages.create(model=model, max_tokens=1200, system=system,
                                         messages=[{"role": "user", "content": prompt}])
            return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text").strip(), ""
        if backend == "openai-compat":
            from .research import run_openai_compat
            return run_openai_compat(system + "\n\n" + prompt, dict(cfg, model=model), timeout).strip(), ""
        if backend == "claude-cli":
            exe = shutil.which("claude")
            if not exe:
                return "", "claude CLI not found on PATH"
            args = [exe, "-p", "--output-format", "text"]
            if model:
                args += ["--model", model]
            args += ["--system-prompt", system]
            res = subprocess.run(args, input=prompt, capture_output=True, text=True, timeout=timeout)
            if res.returncode != 0:
                return "", f"claude CLI exit {res.returncode}: {res.stderr[:120]}"
            return res.stdout.strip(), ""
        return "", "llm disabled"
    except Exception as e:  # noqa
        return "", f"{type(e).__name__}: {str(e)[:120]}"


def _fake_card(packet: dict, cfg: dict) -> str:
    """Deterministic stand-in for queue/latency testing (backend: fake)."""
    time.sleep(float(cfg.get("fake_delay_s", 2.0)))
    lines = []
    for letter, r in zip("ABCDE", packet["candidates_ranked_by_adj"][:5]):
        lines.append(f"{letter}  {r['name']}  {r['pos']} T{r['tier']}  vorp {r['vorp']}  surv {r['surv']:.0f}%  "
                     f"bye {r['bye']}  stack -  need {r['need']}  via VALUE")
        lines.append("   + (fake backend)")
        lines.append("   - (fake backend)")
    lines.append("IF A gone -> B")
    dp = packet["data_pick"]["name"]
    lines.append(f"DATA {dp}: (fake backend)")
    lines.append(f"SLEEPER {dp}: (fake backend)")
    return "\n".join(lines)


def llm_card(packet: dict, cfg: dict) -> tuple[str, str]:
    """Returns (card_text, status). status = 'llm' | 'fallback:<reason>'."""
    backend, _, _ = _resolve(cfg)
    if backend == "none":
        return "", "fallback:llm disabled"
    prompt = (
        "PACKET (JSON):\n" + json.dumps(packet, indent=1, default=str) +
        "\n\nRules:\n"
        "1. Pick exactly five candidates from candidates_ranked_by_adj (all of them if fewer). They are PEERS "
        "that each pass a criterion, not a ranking - B-E are acceptable picks, not runners-up. Tag each with "
        "'via <criterion>'.\n"
        "2. Copy vorp/surv/bye/stack/need from the packet verbatim.\n"
        "3. Consider: open starting slots and backup slots, tier cliffs (tier_empty_p, cliff, tiers_remaining), "
        "wait_cost vs surv, risk 3+ (say why: role/inj/tags), bye_team_count 2+ (a warning, not a veto - weigh "
        "upside against the bye overload and say so), stack with my QB, anti-crossovers, strategy_notes, and what "
        "opponents need before my next pick.\n"
        "4. Add a '! ' warning line under any option with bye_team_count >= 2 or risk >= 3.\n"
        "5. Close with the DATA line (echo data_pick verbatim) and the SLEEPER line (your considered call; may "
        "equal DATA; a deviation must cite its criterion or strategy note; best_available_by_pos is eligible only "
        "with a strategy_notes or cliff justification).\n"
        "6. Output ONLY in this exact format:\n" + CARD_FORMAT
    )
    if backend == "fake":
        text = _fake_card(packet, cfg)
    else:
        text, err = _call_backend(SYSTEM, prompt, cfg)
        if err:
            return "", f"fallback:{err}"
    ok, why = _validate(text, packet)
    if not ok:
        return text, f"fallback:{why}"
    return text, "llm"


def post_pick_card(packet: dict, cfg: dict) -> tuple[str, str]:
    """Team-state analysis right after my pick. Returns (text, status)."""
    backend, _, _ = _resolve(cfg)
    if backend == "none":
        return "", "fallback:llm disabled"
    prompt = "PACKET (JSON):\n" + json.dumps(packet, indent=1, default=str) + \
             "\n\nReview the roster per the system instructions. Output the lines only."
    if backend == "fake":
        time.sleep(float(cfg.get("fake_delay_s", 2.0)))
        return "ON PLAN (fake backend)", "llm"
    text, err = _call_backend(POSTPICK_SYSTEM, prompt, cfg)
    if err:
        return "", f"fallback:{err}"
    return text, "llm"


def write_card(out_dir: str, header: str, det: str, llm: str | None, status: str):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "card_latest.md"), "w") as f:
        f.write(f"# {header}\n\n## Harness (deterministic)\n```\n{det}\n```\n")
        if llm:
            f.write(f"\n## Claude ({status})\n```\n{llm}\n```\n")
