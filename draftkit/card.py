"""The decision card: deterministic top-3 always; LLM-adjudicated version when available."""
from __future__ import annotations

import json
import os
import shutil
import subprocess

import pandas as pd

from .data import norm_name

CARD_FORMAT = """A  <name>  <pos> T<tier>  vorp <n>  surv <n>%  bye <n> <ok|XX>  stack <+n|—>  need <tag>
   + <one line, the strongest reason>
   - <one line, the strongest objection>
B  ...
   + ...
   - ...
C  ...
   + ...
   - ...
IF A gone -> <B or C>: <one clause>
IF A,B gone -> <C or another candidate>: <one clause>"""

SYSTEM = """You are adjudicating a fantasy football draft pick. You receive a packet of precomputed numbers for a closed set of candidates. You do not compute, estimate, or recall anything about players beyond the packet — every number you print must be copied from it. You choose three candidates from the packet, rank them, and explain each in exactly one + line and one - line. Your output is displayed to a human who has under three minutes to click. Be terse. No preamble, no closing remarks, no markdown headers."""


def _fmt_line(letter: str, r: pd.Series) -> str:
    bye = f"bye {r['bye']} {'XX' if r['bye_cost'] > 0 else 'ok'}"
    stack = f"stack +{r['stack']:.0f}" if r["stack"] > 0 else "stack —"
    return (f"{letter}  {r['name']:<22} {r['pos']} T{r['tier']}  vorp {r['vorp']:>5.1f}  surv {r['surv']*100:>3.0f}%  "
            f"{bye}  {stack}  need {r['need']}")


def deterministic_card(cands: pd.DataFrame, tiers: pd.DataFrame, picks_until_next: int | None) -> str:
    if cands.empty:
        return "(no candidates)"
    lines = []
    top = cands.head(3).reset_index(drop=True)
    for i, r in top.iterrows():
        letter = "ABC"[i]
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
        if r["bye_cost"] > 0:
            minus.append(f"bye {r['bye']} collides with {r['bye_with']} (-{r['bye_cost']:.0f})")
        if r["anti"] > 0:
            minus.append(f"cannibalizes {r['anti_with']} (-{r['anti']:.0f})")
        if r["risk"] >= 3:
            minus.append(f"risk {r['risk']}, role {r['role']}, inj {r['inj']}")
        if r["surv"] >= 0.6 and r["wait_cost"] <= 2:
            minus.append(f"likely still here next turn ({r['surv']*100:.0f}%) — can wait")
        lines.append("   + " + (plus[0] if plus else f"best adjusted value ({r['adj']:.1f})"))
        lines.append("   - " + (minus[0] if minus else f"risk {r['risk']}, {r['role']}"))
    if len(top) >= 2:
        lines.append(f"IF A gone -> B ({top.loc[1,'name']})")
    if len(top) >= 3:
        lines.append(f"IF A,B gone -> C ({top.loc[2,'name']})")
    return "\n".join(lines)


def build_packet(summary: dict, cands: pd.DataFrame, roster: pd.DataFrame, tiers: pd.DataFrame,
                 opp_needs: dict[int, str], strategy_notes: str, horizon_pick: int | None) -> dict:
    keep = ["name", "pos", "team", "bye", "tier", "proj", "vorp", "adp", "surv", "wait_cost", "need", "need_pts",
            "bye_cost", "bye_with", "stack", "stack_with", "anti", "anti_with", "risk", "role", "inj", "tags",
            "tier_empty_p", "adj"]
    c = cands[keep].copy()
    c["surv"] = (c["surv"] * 100).round(0)
    c["tier_empty_p"] = (c["tier_empty_p"] * 100).round(0)
    tl = tiers[tiers["left"] > 0].copy()
    tl["cliff"] = tl["cliff"].round(1)
    return {
        "draft": {k: summary.get(k) for k in ["round", "pick_in_round", "picks_until_me", "on_clock_is_me"]},
        "my_next_pick_after_this": horizon_pick,
        "my_roster": roster[["name", "pos", "team", "bye", "tier", "proj"]].to_dict("records") if len(roster) else [],
        "candidates_ranked_by_adj": c.to_dict("records"),
        "tiers_remaining": tl.to_dict("records"),
        "opponent_needs_before_my_next_pick": opp_needs,
        "strategy_notes": strategy_notes,
        "column_glossary": {
            "vorp": "season pts over positional replacement (league-specific)",
            "surv": "% of sims in which this player is still available at my next pick",
            "wait_cost": "vorp minus expected best vorp at this position at my next pick; what waiting costs",
            "need": "STARTER=open fixed slot, FLEX=open flex slot, BENCH=depth only, CAPPED=too many at pos",
            "bye_cost": "pts lost from same-slot bye collisions with my roster",
            "stack": "pts bonus for correlation with my QB/receivers",
            "anti": "pts penalty for cannibalizing a player I roster",
            "tier_empty_p": "% of sims in which this player's tier is gone by my next pick",
            "adj": "deterministic combined score; the harness's own ranking",
        },
    }


def _validate(text: str, cands: pd.DataFrame) -> tuple[bool, str]:
    names = {norm_name(n) for n in cands["name"]}
    found = 0
    for line in text.splitlines():
        s = line.strip()
        if len(s) > 2 and s[0] in "ABC" and s[1] == " ":
            body = norm_name(s[1:])
            if any(n and n in body for n in names):
                found += 1
            else:
                return False, f"option line names a player outside the candidate set: {s[:60]}"
    if found < 2:
        return False, "fewer than two option lines parsed"
    return True, ""


def llm_card(packet: dict, cfg: dict) -> tuple[str, str]:
    """Returns (card_text, status). status = 'llm' | 'fallback:<reason>'."""
    backend = (cfg or {}).get("backend", "none")
    model = (cfg or {}).get("model", "")
    prompt = (
        "PACKET (JSON):\n" + json.dumps(packet, indent=1, default=str) +
        "\n\nRules:\n"
        "1. Pick exactly three candidates from candidates_ranked_by_adj. You may reorder them; if you demote the "
        "harness's #1, your + line for your A must say why in the numbers.\n"
        "2. Copy vorp/surv/bye/stack/need from the packet verbatim.\n"
        "3. Consider: open starting slots, tier cliffs (tier_empty_p, tiers_remaining), wait_cost vs surv, bye "
        "collisions, stack with my QB, anti-crossovers, strategy_notes, and what opponents need before my next pick.\n"
        "4. Output ONLY in this exact format:\n" + CARD_FORMAT
    )
    try:
        if backend == "anthropic":
            import anthropic  # noqa
            client = anthropic.Anthropic()
            msg = client.messages.create(model=model, max_tokens=700, system=SYSTEM,
                                         messages=[{"role": "user", "content": prompt}])
            text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text").strip()
        elif backend == "openai-compat":
            from .research import run_openai_compat
            text = run_openai_compat(SYSTEM + "\n\n" + prompt, cfg, float(cfg.get("timeout_s", 90))).strip()
        elif backend == "claude-cli":
            exe = shutil.which("claude")
            if not exe:
                return "", "fallback:claude CLI not found on PATH"
            args = [exe, "-p", "--output-format", "text"]
            if model:
                args += ["--model", model]
            args += ["--system-prompt", SYSTEM]
            res = subprocess.run(args, input=prompt, capture_output=True, text=True, timeout=float(cfg.get("timeout_s", 90)))
            if res.returncode != 0:
                return "", f"fallback:claude CLI exit {res.returncode}: {res.stderr[:120]}"
            text = res.stdout.strip()
        else:
            return "", "fallback:llm disabled"
    except Exception as e:  # noqa
        return "", f"fallback:{type(e).__name__}: {str(e)[:120]}"
    ok, why = _validate(text, pd.DataFrame(packet["candidates_ranked_by_adj"]))
    if not ok:
        return text, f"fallback:{why}"
    return text, "llm"


def write_card(out_dir: str, header: str, det: str, llm: str | None, status: str):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "card_latest.md"), "w") as f:
        f.write(f"# {header}\n\n## Harness (deterministic)\n```\n{det}\n```\n")
        if llm:
            f.write(f"\n## Claude ({status})\n```\n{llm}\n```\n")
