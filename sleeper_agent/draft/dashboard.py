"""Terminal dashboard"""
from __future__ import annotations

import time

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ..core.data import POS


class Dashboard:
    def __init__(self, start: bool = True):
        self.console = Console()
        self.live = Live(console=self.console, refresh_per_second=4, screen=True)
        if start:
            self.live.start()

    def stop(self):
        self.live.stop()

    def render(self, eng):
        self.live.update(self.build(eng))

    def build(self, eng):
        s = eng.state.summary()
        clock = s["on_clock_since_s"]
        left = eng.meta.pick_timer - clock if clock is not None else None
        me = " [bold red]>>> YOU ARE ON THE CLOCK <<<[/]" if s["on_clock_is_me"] else ""
        eta = f"{s['eta_s']/60:.1f} min" if s.get("eta_s") is not None else "-"
        hdr = Text.from_markup(
            f"[bold]R{s['round']}.{s['pick_in_round']}[/]  pick #{s['last_pick_no']+1}  on clock: slot {s['on_clock_slot']}{me}   "
            f"picks until me: [bold]{s['picks_until_me']}[/]  eta ~{eta}   "
            f"autodraft slots: {s['autodraft_slots'] or '-'}"
        )
        # -- appended, not interpolated: a model name may contain brackets rich would read as markup
        if eng.llm_cfg.get("backend", "none") not in ("none", "fake") and eng.llm_cfg.get("draft_model"):
            hdr.append(f"   {eng.llm_cfg['draft_model']}")

        # my roster
        rt = Table(title="My roster", expand=True, show_edge=False, pad_edge=False)
        for c in ["Pos", "Name", "Bye"]:
            rt.add_column(c)
        roster = eng.my_roster()
        bye_counts = roster["bye"].astype(int).value_counts().to_dict() if len(roster) else {}
        for _, r in roster.sort_values(["pos", "proj"], ascending=[True, False]).iterrows():
            bye = int(r["bye"])
            bye_s = f"[bold red]{bye}[/]" if bye_counts.get(bye, 0) >= 3 else str(bye)
            rt.add_row(r["pos"], r["name"], bye_s)
        if eng.cands is not None:
            fo = eng.cands.attrs.get("fixed_open", {})
            fl = eng.cands.attrs.get("flex_open", 0)
            bo = eng.cands.attrs.get("backup_open", {})
            cap = "open: " + " ".join(f"{p}x{n}" for p, n in fo.items() if n) + f" FLEXx{fl}"
            if any(bo.values()):
                cap += "  bkup: " + " ".join(f"{p}x{n}" for p, n in bo.items() if n)
            rt.caption = cap

        # tiers
        tt = Table(title="Tiers left (pos T: left/total, cliff)", expand=True, show_edge=False, pad_edge=False)
        tpos = [p for p in POS if getattr(eng, "tiers", None) is not None and (eng.tiers.pos == p).any()] or POS[:4]
        for c in tpos:
            tt.add_column(c)
        if getattr(eng, "tiers", None) is not None:
            cols = {}
            for p in tpos:
                sub = eng.tiers[(eng.tiers.pos == p) & (eng.tiers.left > 0)].head(5)
                cols[p] = [f"T{int(r.tier)}: {int(r.left)}/{int(r.total)}  ↓{r.cliff:.0f}" if r.cliff == r.cliff else f"T{int(r.tier)}: {int(r.left)}/{int(r.total)}"
                           for r in sub.itertuples()]
            n = max(len(v) for v in cols.values()) if cols else 0
            for i in range(n):
                tt.add_row(*[cols[p][i] if i < len(cols[p]) else "" for p in tpos])

        # candidates
        ct = Table(title="Candidates (deterministic ranking)", expand=True, show_edge=True, border_style="white", pad_edge=False)
        for c in ["#", "Name", "Pos", "T", "VORP", "surv", "wait", "need", "bye", "stack", "anti", "risk", "adj"]:
            ct.add_column(c, justify="right" if c not in ("Name", "Pos", "need") else "left")
        if eng.cands is not None:
            for i, r in enumerate(eng.cands.itertuples(), 1):
                bye = f"{r.bye}{' XX' if r.bye_team_count >= 2 else ''}"
                ct.add_row(str(i), r.name, r.pos, str(r.tier), f"{r.vorp:.1f}", f"{r.surv*100:.0f}%", f"{r.wait_cost:.0f}",
                           r.need, bye, f"+{r.stack:.0f}" if r.stack > 0 else "", f"-{r.anti:.0f}" if r.anti > 0 else "",
                           str(r.risk), f"{r.adj:.1f}")

        det = Panel(eng.det_card or "", title="Est. Best Available", border_style="cyan")
        # -- title names WHAT is on screen (which model output); the appendage reports the
        #    background job: a live counter while one runs, an error flag when the last one failed.
        #    A failed call leaves the previous card on screen, so the pick stamp and STALE marker
        #    keep advice for an earlier pick from reading as advice for the one on the clock.
        job_label = {"precard": "Pre-Pick", "card": "Sleeper Pick", "postpick": "Post-Pick"}
        nxt = eng.state.next_my_pick()
        stale = (eng.llm_kind in ("precard", "card") and eng.llm_pick is not None
                 and nxt is not None and eng.llm_pick != nxt)
        lt = job_label.get(eng.llm_kind, "Sleeper Pick")
        if eng.llm_pick is not None and eng.llm_text:
            lt += f" (pick {eng.llm_pick}{', STALE' if stale else ''})"
        if eng.llm_running is not None and eng.llm_started is not None:
            lt += f" (running {job_label.get(eng.llm_running, eng.llm_running)} {time.time() - eng.llm_started:.0f}s)"
        elif eng.llm_error is not None:
            lt += f" ERROR {job_label.get(eng.llm_error, eng.llm_error)} - see log"
        # -- Text(), not the bare string: Panel parses a str title as markup and would eat any
        #    bracketed fragment as a style tag, silently dropping it from the border
        llm = Panel(eng.llm_text or ("(llm disabled)" if eng.llm_cfg.get("backend", "none") == "none" else "(waiting)"),
                    title=Text(lt),
                    border_style="red" if stale else ("yellow" if eng.llm_error or not eng.llm_text else "green"))

        # on-the-clock box (doubles as connection status)
        if eng.conn_state != "ok":
            msg = "reconnecting…" if eng.conn_state == "retry10" else "CONNECTION LOST\nclose and restart\nthe program"
            clk = Panel(Text(msg, style="bold red"), title="connection", border_style="red")
        else:
            slot = s["on_clock_slot"]
            if slot is None:
                clk = Panel("draft complete", title="On the clock", border_style="dim")
            else:
                name = eng.meta.slot_names.get(slot, f"slot {slot}")
                lat = eng.state.latency.get(slot, [])
                med = f"{eng.state.median_latency(slot):.0f}s" if lat else "-"
                avg = f"{eng.state.mean_latency(slot):.0f}s" if lat else "-"
                auto = " [bold yellow]AUTO[/]" if eng.state.is_autodraft(slot) else ""
                body = Text.from_markup(
                    f"slot {slot}  [bold]{name}[/]{auto}\n"
                    f"med {med}  avg {avg}\n"
                    f"clock: {'-' if left is None else f'{left:.0f}s'}"
                )
                clk = Panel(body, title="On the clock", border_style="magenta" if s["on_clock_is_me"] else "dim")

        layout = Layout()
        layout.split_column(Layout(Panel(hdr), size=3), Layout(name="body"))
        layout["body"].split_row(Layout(name="left", ratio=1), Layout(name="right", ratio=3))
        # -- roster is sized to its own rows so the tiers table rides up under it rather than
        #    floating at the column's foot; the trailing pad soaks the leftover height. The ceiling
        #    keeps tiers and the clock box on screen once the roster fills out in the late rounds.
        #    The pad holds a space, not an empty string: Layout falls back to drawing a placeholder
        #    box on a falsy renderable.
        rsize = max(6, min(len(roster) + 6, self.console.size.height - 3 - 14))
        layout["left"].split_column(Layout(rt, size=rsize), Layout(tt, size=9),
                                    Layout(clk, size=5), Layout(Text(" "), name="pad"))
        layout["right"].split_column(Layout(ct, size=15), Layout(det, size=8), Layout(llm))
        return layout
