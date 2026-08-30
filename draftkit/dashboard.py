"""Terminal dashboard (rich)."""
from __future__ import annotations

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


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
            f"picks until me: [bold]{s['picks_until_me']}[/]  eta ~{eta}   clock: {'-' if left is None else f'{left:.0f}s'}   "
            f"autodraft slots: {s['autodraft_slots'] or '-'}"
        )

        # my roster
        rt = Table(title="My roster", expand=True, show_edge=False, pad_edge=False)
        for c in ["Pos", "Name", "Team", "Bye", "T", "Proj"]:
            rt.add_column(c)
        roster = eng.my_roster()
        for _, r in roster.sort_values(["pos", "proj"], ascending=[True, False]).iterrows():
            rt.add_row(r["pos"], r["name"], r["team"], str(int(r["bye"])), str(int(r["tier"])), f"{r['proj']:.0f}")
        if eng.cands is not None:
            fo = eng.cands.attrs.get("fixed_open", {})
            fl = eng.cands.attrs.get("flex_open", 0)
            rt.caption = "open: " + " ".join(f"{p}x{n}" for p, n in fo.items() if n) + f" FLEXx{fl}"

        # tiers
        tt = Table(title="Tiers left (pos T: left/total, cliff)", expand=True, show_edge=False, pad_edge=False)
        for c in ["QB", "RB", "WR", "TE"]:
            tt.add_column(c)
        if getattr(eng, "tiers", None) is not None:
            cols = {}
            for p in ["QB", "RB", "WR", "TE"]:
                sub = eng.tiers[(eng.tiers.pos == p) & (eng.tiers.left > 0)].head(5)
                cols[p] = [f"T{int(r.tier)}: {int(r.left)}/{int(r.total)}  ↓{r.cliff:.0f}" if r.cliff == r.cliff else f"T{int(r.tier)}: {int(r.left)}/{int(r.total)}"
                           for r in sub.itertuples()]
            n = max(len(v) for v in cols.values()) if cols else 0
            for i in range(n):
                tt.add_row(*[cols[p][i] if i < len(cols[p]) else "" for p in ["QB", "RB", "WR", "TE"]])

        # candidates
        ct = Table(title="Candidates (harness ranking)", expand=True, show_edge=False, pad_edge=False)
        for c in ["#", "Name", "Pos", "T", "VORP", "surv", "wait", "need", "bye", "stack", "anti", "risk", "adj"]:
            ct.add_column(c, justify="right" if c not in ("Name", "Pos", "need") else "left")
        if eng.cands is not None:
            for i, r in enumerate(eng.cands.itertuples(), 1):
                bye = f"{r.bye}{' XX' if r.bye_cost > 0 else ''}"
                ct.add_row(str(i), r.name, r.pos, str(r.tier), f"{r.vorp:.1f}", f"{r.surv*100:.0f}%", f"{r.wait_cost:.0f}",
                           r.need, bye, f"+{r.stack:.0f}" if r.stack > 0 else "", f"-{r.anti:.0f}" if r.anti > 0 else "",
                           str(r.risk), f"{r.adj:.1f}")

        det = Panel(eng.det_card or "", title="Harness card", border_style="cyan")
        llm_title = f"Claude card [{eng.llm_status}]" if eng.llm_status else "Claude card"
        llm = Panel(eng.llm_text or ("(llm disabled)" if eng.cfg.get("llm", {}).get("backend", "none") == "none" else "(waiting for your turn)" if not s["on_clock_is_me"] else "(running...)"),
                    title=llm_title, border_style="green" if eng.llm_status == "llm" else "yellow")
        msgs = Panel("\n".join(eng.messages[-8:]), title="Log", border_style="dim")

        layout = Layout()
        layout.split_column(Layout(Panel(hdr), size=3), Layout(name="body"), Layout(msgs, size=10))
        layout["body"].split_row(Layout(name="left", ratio=2), Layout(name="right", ratio=3))
        layout["left"].split_column(Layout(rt), Layout(tt, size=9))
        layout["right"].split_column(Layout(ct, size=13), Layout(det), Layout(llm))
        return layout
