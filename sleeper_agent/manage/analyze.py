"""The weekly manage report under four headings: Moves, Lineup, Injuries, Passing on. The engine writes it from
the packet alone; a model pass may rewrite the four sections, kept only if `validate` passes. Writes
out/manage/w<N>/report.md, packet.json and, for a rejected reply, rejected.md; a replay writes to w<N>-replay.

    python3 -m sleeper_agent manage [--week N | --as-of-week N] [--no-llm] [--watch NAME] [--note TEXT] [--model M]
"""
from __future__ import annotations

import json
import os
import re
import unicodedata

from ..data import norm_name
from ..league import open_league
from ..llm import chat, resolve
from .packet import build_packet
from .replay import open_replay

HEADINGS = ("Moves", "Lineup", "Injuries", "Passing on")
POSITIONS = {"QB", "RB", "WR", "TE", "K"}
SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}

SYSTEM = """You are the analyst for one week of a fantasy football manager's in-season league. You receive a packet of precomputed facts as JSON, and its glossary defines every field. You do not compute, estimate or recall anything about players beyond the packet. Every number you print is copied from the packet, and you may name only players listed in closed_set. A player named only in a note or a news row may be discussed, never added.

Reason across sections before you write. A free reserve slot changes which drop an add needs. A bye hole changes who starts and which add is worth making. A note from the manager can override a default. You recommend; the manager makes every move in Sleeper.

Write Markdown under exactly these four headings, in this order: ## Moves, ## Lineup, ## Injuries, ## Passing on. Under Moves, write one list item per move, each opening with exactly one bold action in one of these forms: **Add X, drop Y**, **Add X**, **Move X to IR**, **Activate X**, **Hold**. X and Y are full player names spelled as in the packet. An added player comes from adds; a dropped, reserved or activated player comes from roster. Follow each action with one or two sentences giving the reason and the packet numbers behind it. Write **Hold** when no move helps. Under Lineup, say who to start and sit. Under Injuries, say what each flag means this week. Under Passing on, say why each trending player is skipped. Be terse. No preamble, no closing remarks, no other headings."""

_ITEM = re.compile(r"^(?:[-*+]|\d+[.)])\s+(.*)$")
_BOLD = re.compile(r"^\*\*(.+?)\*\*")
_ACTIONS = (("add_drop", re.compile(r"^add (.+?), drop (.+)$", re.I)),
            ("add", re.compile(r"^add (.+)$", re.I)),
            ("ir", re.compile(r"^move (.+) to ir$", re.I)),
            ("activate", re.compile(r"^activate (.+)$", re.I)),
            ("hold", re.compile(r"^hold$", re.I)))


def _band(a: dict) -> str:
    lo, hi = a["faab_low"], a["faab_high"]
    if lo is None:
        return ""
    return f"${lo:g}" if lo == hi else f"${lo:g}-${hi:g}"


def _drop(d: dict | None) -> str:
    return f"{d['name']} ({d['pos']}, ros {d['ros']:.1f})" if d else "(open roster spot)"


def _moves(packet: dict) -> list[str]:
    lines = []
    for f in packet["injuries"]:
        if f["kind"] == "ir_open":
            lines.append(f"- MOVE {f['name']} ({f['pos']}, {f['team'] or 'FA'}) to reserve: {f['status']}")
        elif f["kind"] == "ir_cleared":
            lines.append(f"- MOVE {f['name']} ({f['pos']}, {f['team'] or 'FA'}) off reserve: no injury tag")
    first, last = packet["horizon_weeks"]
    if packet["adds"] and packet["adds_assume_moves"]:
        lines.append("- The adds below count the reserve moves above as made.")
    for a in packet["adds"]:
        extra = [f"horizon {a['horizon_proj']:.1f}"]
        if a["trending_adds"]:
            extra.append(f"trending +{a['trending_adds']:,}")
        if _band(a):
            extra.append(f"FAAB {_band(a)}")
        if a["watched"]:
            extra.append("watched")
        lines.append(f"- ADD {a['name']} ({a['pos']}, {a['team'] or 'FA'}) | DROP {_drop(a['drop'])} | "
                     f"{a['gain']:+.1f} over weeks {first}-{last} | " + ", ".join(extra))
        if a["alternatives"]:
            lines.append("  - also: " + ", ".join(f"{_drop(x['drop'])} {x['gain']:+.1f}" for x in a["alternatives"]))
    return lines or ["No moves."]


def _lineup(packet: dict) -> list[str]:
    lu = packet["lineup"]
    if packet.get("replay"):
        return [f"Best lineup {lu['best_total']:.1f} pts; the lineup set that week cannot be recovered.",
                *(f"- START {p['name']} ({p['pos']}) {p['pts']:.1f}" for p in lu["best"])]
    lines = [f"Current {lu['current_total']:.1f} pts, best {lu['best_total']:.1f} pts."]
    if not lu["swaps"]:
        lines.append("Current starters are already optimal.")
    for s in lu["swaps"]:
        start, sit = s["start"], s["sit"]
        sat = f"SIT {sit['name']} ({sit['pos']}) {sit['pts']:.1f}" if sit else "SIT (empty slot)"
        lines.append(f"- START {start['name']} ({start['pos']}) {start['pts']:.1f} | {sat} | {s['delta']:+.1f}")
    return lines


def _passing_on(packet: dict) -> list[str]:
    if packet.get("replay"):
        return ["Trending counts cannot be recovered in a replay; no player is passed on."]
    lines = [f"- {p['name']} ({p['pos']}, {p['team'] or 'FA'}): "
             + (f"trending +{p['trending_adds']:,}, " if p["trending_adds"] else "")
             + f"best gain {p['best_gain']:+.1f}, horizon {p['horizon_proj']:.1f}" for p in packet["passing_on"]]
    return lines or ["No trending free agent outside the adds."]


def _header(packet: dict, note: str = "") -> list[str]:
    res = packet["research"]
    faab = packet["faab"]
    dated = f" ({res['date']})" if res["date"] else ""
    replay = packet.get("replay")
    research = "not used in a replay" if replay else f"{res['set'] or 'none'}{dated}"
    record = "" if replay else f"Record {packet['record']}, "
    lines = [f"# {packet['league']}, week {packet['week']}" + (" (replay)" if replay else ""), "",
             f"{record}FAAB ${faab['left']:g} of ${faab['budget']:g}. Research set: {research}."]
    if replay:
        lines += ["", f"> REPLAY as of week {replay['as_of_week']}: rosters and FAAB rebuilt to {replay['cutoff']}. "
                      "Past injury statuses, lineups and reserve lists cannot be recovered; every player counts as "
                      "healthy, no lineup is set and reserve is empty."]
    if res["warning"]:
        lines += ["", f"> WARNING: {res['warning']}"]
    if note:
        lines += ["", f"> {note}"]
    return lines


def _body(packet: dict, level: int = 2) -> list[str]:
    sections = [("Moves", _moves(packet)), ("Lineup", _lineup(packet)),
                ("Injuries", [f"- {f['text']}" for f in packet["injuries"]] or ["No injury flags."]),
                ("Passing on", _passing_on(packet))]
    lines = []
    for title, body in sections:
        lines += ["", f"{'#' * level} {title}", "", *body]
    return lines


def deterministic_report(packet: dict, note: str = "") -> str:
    return "\n".join(_header(packet, note) + _body(packet)) + "\n"


def compose_report(packet: dict, text: str, status: str) -> str:
    """The model's four sections over the engine's, demoted under `## Engine numbers`; on a fallback, the
    engine report with the reason in its header."""
    if status != "llm":
        reason = status.split(":", 1)[-1]
        return deterministic_report(packet, f"Engine report only; no model analysis: {reason}.")
    reply = text.strip().splitlines()
    start = next(i for i, line in enumerate(reply) if line.strip().startswith("## "))
    note = "The four sections below are the model's analysis, checked against the packet."
    lines = _header(packet, note) + [""] + reply[start:] + ["", "## Engine numbers"] + _body(packet, 3)
    return "\n".join(lines) + "\n"


def _fold(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    return " ".join(re.findall(r"[a-z0-9]+", s.replace(".", "").replace("'", "")))


def _name_fold(name: str) -> str:
    words = _fold(name).split()
    if len(words) > 2 and words[-1] in SUFFIXES:
        words = words[:-1]
    return " ".join(words)


def _sections(text: str) -> list[tuple[str, list[str]]]:
    out: list[tuple[str, list[str]]] = []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("## "):
            out.append((s[3:].strip(), []))
        elif out:
            out[-1][1].append(line)
    return out


def _parse_action(bold: str) -> tuple[str, list[str]] | None:
    s = re.sub(r"\s*\([^)]*\)", "", bold).strip().rstrip(".:;!").strip()
    for kind, rx in _ACTIONS:
        m = rx.match(s)
        if m:
            return kind, [g.strip() for g in m.groups()]
    return None


def _outside_names(packet: dict, players: dict) -> dict[str, str]:
    closed = {norm_name(n) for n in packet["closed_set"]}
    # -- a player the news notes or the manager's notes name may be discussed
    quoted = [str(r.get("note") or "") for r in packet.get("news") or []] + [str(n) for n in packet.get("notes") or []]
    allowed = f" {_fold(' | '.join(quoted))} "
    out = {}
    for m in players.values():
        name = m.get("name") or ""
        if m.get("pos") not in POSITIONS or not name or norm_name(name) in closed:
            continue
        folded = _name_fold(name)
        if " " in folded and f" {folded} " not in allowed:
            out[folded] = name
    return out


def validate(text: str, packet: dict, players: dict) -> tuple[bool, str]:
    """Checks the reply's headings, its Moves actions against adds and my roster, and that no QB, RB, WR, TE
    or K in the Sleeper table is named in full unless `closed_set`, a news note or a manager note names him."""
    secs = _sections(text)
    heads = [h for h, _ in secs]
    for h in HEADINGS:
        if h not in heads:
            return False, f"heading missing: {h}"
    if tuple(heads) != HEADINGS:
        return False, f"headings out of order: {', '.join(heads)}"
    roster = {norm_name(r["name"]) for r in packet["roster"]}
    adds = {norm_name(a["name"]) for a in packet["adds"]}
    items = [m.group(1) for line in dict(secs)["Moves"] if (m := _ITEM.match(line))]
    if not items:
        return False, "Moves holds no list item"
    for item in items:
        bold = _BOLD.match(item)
        parsed = _parse_action(bold.group(1)) if bold else None
        if parsed is None:
            return False, f"Moves item opens without a known bold action: {item[:60]}"
        kind, names = parsed
        if kind in ("add", "add_drop") and norm_name(names[0]) not in adds:
            return False, f"added player outside adds: {names[0]}"
        mine = names[1:] if kind == "add_drop" else names if kind in ("ir", "activate") else []
        for n in mine:
            if norm_name(n) not in roster:
                return False, f"player off my roster: {n}"
    folded = f" {_fold(text)} "
    for name_fold, name in _outside_names(packet, players).items():
        if f" {name_fold} " in folded:
            return False, f"names a player outside closed_set: {name}"
    return True, ""


def _fake_reply(packet: dict) -> str:
    moves = []
    for f in packet["injuries"]:
        if f["kind"] == "ir_open":
            moves.append(f"- **Move {f['name']} to IR**: {f['status']} opens a reserve slot.")
        elif f["kind"] == "ir_cleared":
            moves.append(f"- **Activate {f['name']}**: no injury tag.")
    for a in packet["adds"][:1]:
        act = f"Add {a['name']}, drop {a['drop']['name']}" if a["drop"] else f"Add {a['name']}"
        moves.append(f"- **{act}**: gain {a['gain']:+.1f}.")
    lineup = [f"- Start {s['start']['name']}" + (f", sit {s['sit']['name']}" if s["sit"] else "")
              for s in packet["lineup"]["swaps"]]
    lines = ["## Moves", "", *(moves or ["- **Hold**: no move gains points."]),
             "", "## Lineup", "", *(lineup or ["Current starters are already optimal."]),
             "", "## Injuries", "", *([f"- {f['text']}" for f in packet["injuries"]] or ["No injury flags."]),
             "", "## Passing on", "",
             *([f"- {p['name']}: best gain {p['best_gain']:+.1f}." for p in packet["passing_on"]] or ["Nobody."])]
    return "\n".join(lines)


def llm_report(packet: dict, cfg: dict, players: dict, model: str = "") -> tuple[str, str]:
    """One `manage` chat call over the packet. `cfg` is the `llm` block; `players` the Sleeper table.
    Returns (text, status), status `llm` or `fallback:<reason>`."""
    backend = resolve(cfg, "manage")[0]
    if backend == "none":
        return "", "fallback:llm disabled"
    if backend == "fake":
        text = _fake_reply(packet)
    else:
        prompt = ("PACKET (JSON):\n" + json.dumps(packet, indent=1, default=str) +
                  "\n\nWrite this week's report per the system instructions.")
        text, err = chat(SYSTEM, [{"role": "user", "content": prompt}], dict(cfg, manage_model=model) if model else cfg,
                         phase="manage")
        if err:
            return "", f"fallback:{err}"
    ok, why = validate(text, packet, players)
    return (text, "llm") if ok else (text, f"fallback:{why}")


def cmd_manage(cfg, args):
    as_of = args.as_of_week
    if as_of:
        api, snap, replay = open_replay(cfg, as_of)
        week = as_of
    else:
        (api, snap), replay = open_league(cfg), None
        week = args.week or snap.week
    packet = build_packet(api, snap, cfg, week, notes=args.note or (), watch=args.watch or (), replay=replay)
    llm_cfg = cfg.get("llm") or {}
    if args.no_llm:
        text, status = "", "fallback:llm disabled"
    else:
        text, status = llm_report(packet, llm_cfg, api.players(), model=args.model)
    report = compose_report(packet, text, status)
    print(report)
    out = os.path.join(cfg["out_dir"], "manage", f"w{week}-replay" if replay else f"w{week}")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "report.md"), "w") as f:
        f.write(report)
    with open(os.path.join(out, "packet.json"), "w") as f:
        json.dump(packet, f, indent=1)
    rejected = os.path.join(out, "rejected.md")
    if status != "llm" and text:
        with open(rejected, "w") as f:
            f.write(f"<!-- {status} -->\n\n{text}\n")
    elif os.path.exists(rejected):
        os.remove(rejected)
    print(f"model: {status}")
    print(f"wrote {os.path.join(out, 'report.md')}, {os.path.join(out, 'packet.json')}")
