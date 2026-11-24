# ball-knower

A Sleeper Sports snake-draft assistant and agentic research workflow. It researches, follows, and suggests, you make the pick.

An agent workflow shards research on a *per-team* or *per-division* basis of all players in the top-140 ADP. Each agent researches and analyzes recent player news, injuries, opinions, and history, team staffing developments, as well as per-team personnel (committee backfields, shared targets, pass vs. run-heavy schemes) to produce research tables that combine base stats (VORP, bye weeks, etc) with computed metrics.

At draft time, the system uses a deterministic scoring algorithm based on VORP and draft-related metrics and Monte Carlo simulation of draft outcomes from the *current draft state* to actively re-rank remaining players and present suggestions. At pick time (and optionally a few picks before based on learned time-on-clock of draftees), the system combines the research and derived metrics of the top remaining players with a lightweight model call to provide final model adjudication cards in an A–E format, plus a team-state analysis after each of your picks.

---

## Setup

**Before draft**:

1. `cd ./ball-knower` $\rightarrow$ `pip install -r requirements.txt` $\rightarrow$ set `league_id` (`sleeper.com/leagues/<this_number>/league`) and username in `config.yaml`.
2. `python -m ball_knower setup` - confirms roster/scoring parse, seeds `players_skeleton.csv`, and prints your picks + a settings block.

**Draft day (2+ hrs out)**:

1. Research options:
   - **(best)** `python -m ball_knower research --by team` - Seed players and base stats (bye week, VORP, etc) from Sleeper's projections API base stats. Research agents analyze team and player news *per-team* (4 concurrent, ~35m runtime) to add judgement data (floor/ceil, risk, role, tags, tiers, po_sched), and emit `players.csv`/`stacks.csv`/`anti.csv`. Booyah!
   - **(good)** `python -m ball_knower research` - Seeds base stats, run research agents *per NFL division* (4 concurrent, ~10m) to add judgement data and emit research tables. Best if short on time.
   - **(ok)** `python -m ball_knower fetch` - only generate `data/players_skeleton.csv`. Run `RESEARCH.md` in a LM of your choice with the skeleton attached. Place output under `data\`.
2. Once the draft order posts: `python -m ball_knower script` - your round-by-round expected board.
3. Edit strategy_notes in `config.yaml` for additional rules you want enforced during your draft.
4. (Optional) `python -m ball_knower mock --offline --speed 60` (add --no-llm to skip model card) - dry run on the research tables so the screen and CLI output is familiar.

**Draft Time**:

`python -m ball_knower run`. Leave it running while your draft goes.

When you're on the clock the header turns red. The harness's card is already there, the model output card lands 20–40 s later. Once you pick in Sleeper, the next tick sees your pick and moves on.

## In-season

The tool recommends moves and never writes to Sleeper. You make every add, drop, reserve move and lineup change in Sleeper yourself.

Each week:

1. `python -m ball_knower research --in-season` runs a 7-day news sweep, one agent per NFL team of your roster and the waiver candidates, and writes a refresh set under `data/research/`. The report warns when the latest set is older than `research.max_age_days`.
2. `python -m ball_knower manage` builds the week's packet (lineup, injuries, waiver adds scored over the horizon, usage trends, the research set and its news) and writes `out/manage/w<N>/report.md` and `packet.json`.
3. Read the report, then make the moves you agree with in Sleeper.

The report has four sections: Moves, Lineup, Injuries, Passing on. The engine writes them from the packet alone. Unless `llm.backend` is `none`, one model call then reads the packet and rewrites the four sections, reasoning across them (a free reserve slot, a bye hole, your notes). Each Moves item opens with one bold action: `**Add X, drop Y**`, `**Add X**`, `**Move X to IR**`, `**Activate X**` or `**Hold**`.

The reply is checked before it is used. It needs the four headings in order, every action has to parse, an added player has to be among the packet's adds, a dropped, reserved or activated player has to be on your roster, and no quarterback, back, receiver, tight end or kicker may be named in full unless the packet's `closed_set`, a news note or one of your notes names him. A player named only in a note may be discussed, never added, and a reply may name an opposing team in full. A reply that passes heads the report, with the engine's sections after it under `## Engine numbers`. A reply that fails, or a model error, leaves the engine's report in place with the reason in its header, and the rejected text goes to `out/manage/w<N>/rejected.md`.

| flag | what |
|---|---|
| `--week N` | report week (default: the current NFL week) |
| `--as-of-week N` | replay week N from the league as it stood after week N-1, written to `out/manage/w<N>-replay/`; past injury statuses, lineups and reserve lists cannot be recovered; every player counts as healthy and only the best lineup is shown |
| `--note TEXT` | free text the model reads with the packet, such as "stream a TE over the week 9 bye" (repeatable) |
| `--watch NAME` | include a player; a free agent joins the adds whatever its gain (repeatable) |
| `--no-llm` | skip the model call; the report is the engine's alone |
| `--model M` | override `llm.manage_model` for this run |

## Requirements and Configuration

Requires Python 3.10+.

See `config.yaml`. Documented in [config.md](config.md)

## Commands

All commands read `config.yaml` (`-c other.yaml` to override).

```bash
python -m ball_knower setup    # pull league + draft settings, cache players, print my picks and a
                                  # settings block to paste into RESEARCH_PROMPT.md.
python -m ball_knower script   # Monte Carlo from an empty board: who is likely there at each of
                                  # my picks. --slot N to try a slot before the order is posted.
python -m ball_knower run      # draft day. Dashboard in the terminal; --headless writes files only.
python -m ball_knower mock     # dry run against a simulated draft. --speed 60 finishes in ~3 min,
                                  # --no-llm skips the card call, --slot N, --seed N.
```

---

## Research Tables

- `players.csv`: One row per player with projection, floor/ceiling, tier, ADP, risk, role, injury, bye, and tags from a closed vocabulary.
- `stacks.csv` lists each QB with the pass-catchers whose weekly scores correlate with his, scored 0–100.
- `anti.csv` lists same-team pairs that cannibalize each other with a penalty in season points. The full schema and vocabulary is in `RESEARCH_PROMPT.md`; Every table is keyed by `name`

## Metrics

For the 2026 season, I'm drafting and will be testing the results of the below VORP metric against a range of other metrics.

$$
adj = wait\_w[need\_tag] \,\cdot\, max(wait\_cost, 0) + vorp\_weight \,\cdot\, vorp\_eff + need + stack − anti − risk\_pen
$$

Within a position, this orders by VORP; across positions it orders by what waiting for the player costs, with roster fit as points-denominated adjustments. In terms of drafting, the live LLM reorders only based on remaining players and reasons stated in these numbers.

Bye weeks are **not** in `adj` and surface instead as `bye_team_count` (rostered players sharing a bye) and a `!` warning on the card - see **need** and **Output card** below. `wait_w[need_tag]` reweights based on the need at that position.

**tier / cliff**: Tiers come from the research (gap-based, per position). Cliff is the points lost if you miss a tier and get the top of the next one. The dashboard shows how many players are left in each tier; the MC reports the probability each tier is empty by your next pick (`tier_empty_p`).

**VORP (Value over replacement at position)**: `value − replacement[pos]`. This is the comparison scale across positions. Two adjustments to the classical model:

1. a player who would sit in a flex or bench slot is measured against the *flex* replacement (the highest of the RB/WR/TE replacement levels), because that's who he actually competes with. This stops a TE2 from looking like a top-40 pick.
2. from round 6 onwards, `value` includes the ceiling blend score.

**surv %**: The probability this player is still available at your *next* pick, from the Monte Carlo simulation. When you're on the clock it answers "if I don't take him now, will I get another chance?" - a player at 85% can wait, a player at 5% can't. At the turn (slots 1 and 12) the horizon is your next non-consecutive pick.

**E[best at next] and wait_cost**: For each position, the Monte Carlo estimator reports the expected best VORP still available at your next pick. `wait_cost = vorp − E[best at next][pos]` is what waiting one turn costs you at that position. This is the actual draft-timing criterion: if you'll fill both positions eventually, drafting position P now over Q is right iff `wait_cost(P) > wait_cost(Q)`. A deep position has low wait cost regardless of how good its top player is. `wait_cost` is scaled by `scoring.wait_w[need_tag]` - it only counts for real when you actually intend to fill the slot (STARTER ×1.0, FLEX ×0.8, BACKUP ×0.6, DEPTH ×0.15, CAPPED/K-WAIT ×0).

**need**: a roster-phase *gate* - fill starters, then direct backups, then maximize. Six tags:

- `STARTER` - open fixed slot at their position, `+ need_weight` (×`urgency_factor` once rounds left ≤ open starters + 1).
- `FLEX` - fixed slots at that position filled, an open flex slot accepts them, `+half`.
- `BACKUP` - every (non-K) starter is filled and the position hasn't hit its `backup_targets` count yet: `+ backup_weight × need_weight`. A backup only *covers* a starter if their byes differ - if the candidate's bye collides with a starter (or an already-drafted backup) at that position, the bonus is shaded to `× backup_bye_overlap_factor` and the card notes the weak cover.
- `DEPTH` - past starters and backup targets. While **any** non-K starter slot is still open anywhere on the roster, a DEPTH pick eats a flat `depth_while_starters_open` penalty and its `wait_w` drops to ~0 - this is what stops a 6th WR from outranking an open RB2 on raw value. Once starters are filled but backup targets aren't, the softer `depth_while_backups_open` penalty applies instead. Only once both are satisfied does DEPTH compete on value alone (with the usual `bench_penalty × depth`, halved from `bench_free_round`).
- `CAPPED` - you've hit `pos_caps`: `- capped_multiplier × need_weight`.
- `K-WAIT` - a kicker before the endgame (see **Kickers** below): a flat negative so the harness never burns a mid-round pick on a K, even though K is a fully modeled, VORP'd position.

**bye_team_count / bye warnings**: `bye_cost` is display-only (kept so the `bye`/`bye_with` columns still render) and doesn't affect `adj`. What matters instead is `bye_team_count`: how many players you *already roster team-wide* (not just same position) share the candidate's bye. At `bye_team_count >= 2` (the candidate would be the 3rd+ player on that bye) the card prints a `!` warning - upside is weighed against the pile-up, but the pick is never demoted for it. The one place a bye still moves points is the BACKUP bye-overlap shading above, because a backup that shares its starter's bye isn't actually functioning as cover.

**Kickers**: K is a fully modeled position - fetched, run through research, and VORP'd like QB/RB/WR/TE - but its need bonus is suppressed until the endgame (`K-WAIT`) and only turns on (`STARTER`, urgency-boosted) once `rounds_left <= open_K_slots + 1`. An open K slot never triggers the DEPTH gate above, so it can't block a real roster need mid-draft. DEF remains unmodeled (no VORP, no research row) - but if your league starts a DEF, that slot still consumes a real draft round, and the endgame urgency math (`rounds_left`) is computed net of it so the K/bench squeeze isn't computed as if that round were free. Picks at an unmodeled position (DEF) are tracked as taken but otherwise ignored by opponent-need estimates.

**proj / floor / ceil**: Season points in your league's scoring. Floor $\approx$ 20th percentile outcome, ceil ≈ 80th. In rounds 1–5 the scorer values `proj`; from round 6 it blends toward `ceil` (`scoring.ceiling_blend`), so late picks are judged on upside.

**Replacement level**: Computed from your league's roster, not a rule of thumb. All teams' fixed starting slots are filled greedily by projection, then the flex slots are filled from the best remaining RB/WR/TE, and replacement is the best player left at each position plus `replacement_extra` more (QB and TE get a bigger shift because they're streamable). With two W/R/T flexes this lands around RB30 / WR38 / TE13 / QB15 - pull the exact numbers from `validate`.

**ADP**: Overall pick number from Sleeper's ADP. Drives the survival model. `adp_delta` (your rank − ADP) is not shown directly; the survival model is a better version of the same information.

**stack**: `stack_max_bonus × stack_score/100` when you roster the other half of a `stacks.csv` pair. Same logic: tiebreaker inside a tier, not a reason to reach.

**anti**: Sum of `anti.csv` penalties against players you already roster (committee backfields, shared targets, rushing QB with his own RB).

**risk**: 0–5 from the research rubric, charged at `risk_weight` points per level in rounds 1–6 and half that afterwards.

## Monte Carlo Survival Model

Every tick, for the opponent picks between now and your horizon pick, the system runs Monte Carlo for 2,500 simulated drafts. Each human team draws a personal board `adp + σ(adp)·ε` with `σ = sigma_base + sigma_slope·adp` (ADP standard deviation, ~2 picks at pick 10, ~7 at pick 50, ~13 at pick 100), shifted in $\sigma$ units by roster need (`need_shift`: a position they're capped at is pushed 20σ down, an open starter is 0, bench-only is 1.5σ, urgent open starters late pull up) and by run momentum (`run_shift` $\sigma$ per same-position pick in the last `run_window` picks). They take the lowest score. On top of that, each human slot's style is learned live from its own picks (`mc.slot_tendencies`): the rms of `adp − pick_no` vs what the σ model expects gives a per-slot noise multiplier (reachers get a wider board, chalk drafters a tighter one - a uniform reach bias cancels in the argmin, so bias is expressed as spread), and each slot's pick share by position vs the room becomes a per-position board shift in σ units (`mc.lean_shift`). Both estimates shrink toward neutral until a slot has more than `mc.tendency_prior_picks` picks, and the learned style is shown per slot in the LLM packet's opponent lines (`style: noise x1.3, leans WR`). Autodraft teams are deterministic and take Sleeper's top-ranked available player that fits their roster - a run of only autodrafters before you narrows the tree.

A team is classified autodraft when `min(median, mean)` of its time-on-clock is under `autodraft_latency_s` (default 3.0s), needing ≥3 observations - the mean catches a team that flips autodraft on mid-draft (it drags down fast), while the median guards against a one-off snap pick. Timing prefers the draft object's server-side `last_picked` timestamp over poller observation time whenever a poll sees exactly one new pick; several picks landing in one poll are no longer counted as instant (0s) - each is estimated at half a poll interval, since 0s would poison a human's stats even though it's genuine evidence for autodraft. You can force slots in `mc.autodraft_slots` if you know who's on autopilot.

## Output card

```
A  J. Jacobs   RB  T3  vorp 41.0  surv 18%  bye 7  stack -    need STARTER  via NEED
   + waiting costs ~14 pts at RB (tier closes p=82%, 1 left, cliff 14)
   - risk 3, SHARED
B  ...  via CLIFF
C  ...  via BACKUP
D  ...  via STACK
E  ...  via VALUE
   ! bye 7: would be player #3 on this bye (with J. Chase, T. Higgins)
   ! risk 4: role CONTESTED, inj Q, AGE_CLIFF
IF A gone -> B (...)
DATA J. Jacobs: top adj (41.2) without a 3rd-player bye stack
SLEEPER D. Achane: considered call - cliff at RB is closer than the adj gap suggests
```

The harness writes the deterministic five-option card (A–E, same format the LLM targets) instantly on every board change. Each option is a **peer**, not a ranking - tagged `via <CLIFF|SURV|NEED|BACKUP|STACK|VALUE|SLEEPER>` for the criterion it passes, so B–E read as acceptable picks rather than runners-up. Any option that would be the 3rd+ rostered player on one bye, or carries risk ≥ 3, gets a `!` line naming why - a bye warning never demotes the pick, it's weighed against upside on the line itself. The card closes with two named picks: `DATA <name>` is the harness's own quick answer (highest `adj` that avoids a 3rd-player bye stack, falling back to highest `adj` outright) - a pure function of the packet, so it can't be hallucinated, and the deterministic card and the LLM card agree on it by construction; `SLEEPER <name>` is the model's own considered call given full context (strategy notes, cliffs, stacks, opponent needs) - it often equals DATA, and when it deviates the model must cite the criterion or strategy note driving it.

When you're on the clock (or `llm.at_n_minus` fires early, see below) the same packet (`out/packet_<pick>.json`: candidates with every number above, your roster, open slots, tiers remaining, `cliff` per tier, `bye_team_count`, `best_available_by_pos` - top-2 raw-VORP players per position even outside the ranked list, the harness's `data_pick`, what each opponent before your next pick still needs, `strategy_notes`) goes to the LLM, which returns the same five-option shape. Guardrails: the candidate set is closed (top `top_n` by `adj`), the model is told it may not compute or recall, and its output is parsed - any option naming a player outside the set is rejected and the harness card stands, with the reason in the log; the DATA line must echo the packet's `data_pick` verbatim and the SLEEPER line must name an in-packet or `best_available_by_pos` player.

All LLM calls - the early card, the on-clock card, and the post-pick team-state analysis (below) - run through **one FIFO worker queue**. Jobs are never cancelled once started; a job dequeued after the pick it targeted is already superseded is skipped, but anything already running always finishes. The Claude panel text is never blanked on trigger - the previous card stays up (status shows `[<kind> running…]`) until the new job actually finishes, so suggestions survive across your pick instead of vanishing.

`llm.at_n_minus: true` fires an early card ahead of your pick, at the earliest of two rules: the count rule (`n_minus` spots out) or, from round 4 on (once latency stats exist), the time rule - expected wait to your pick (`min(median, mean)` summed across intervening slots) drops to `eta_trigger_s` or under. A wall of autodrafters ahead of you can trip the time rule several picks before `n_minus` would. `llm.post_pick: true` runs a short team-state analysis call right after you pick - on/off plan vs. strategy notes, top roster holes, any bye pile-up forming - rendered into the same Claude panel and written to `out/teamstate_<pick>.md`; at the snake wheel, a back-to-back pair of your picks gets one analysis, after the second pick, not one after each.

`strategy_notes` in `config.yaml` is free text the LLM sees. Keep it to rules you actually want enforced ("no TE2 before round 12", "stack only inside a tier").

## Dashboard

The terminal UI runs half-screen next to the Sleeper site, so the roster is squished and the site next door shows the rest. No Log panel - `out/log.jsonl` keeps recording every pick, card, and candidate set regardless. Layout: header, then a body split left (roster over tiers) / right (candidates over a bigger Claude card), then a bottom row with the Harness card beside a small on-the-clock box - who's up, their slot name, median/mean pick time, an `AUTO` badge when classified autodraft, and the countdown; it turns red on connection loss. My roster shows only `Pos / Name / Bye`, with a bye cell in bold red when 3+ rostered players share it. On a poll failure the client retries immediately, then again after 10s, then goes to a persistent red "connection lost - close and restart the program" state that keeps quietly retrying every poll interval in the background (and self-clears if the network comes back) - the last good board stays rendered throughout.

## System Files

| path                                               | what                                                                                                     |
| -------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| `RESEARCH_PROMPT.md`                             | The prompt you run 1–2 hours before the draft. Produces the three CSVs.                                 |
| `config.yaml`                                    | League ids, model settings, LLM backend, strategy notes.                                                 |
| `data/players.csv`, `stacks.csv`, `anti.csv` | Research output. Put them here.                                                                          |
| `data/league.json`                               | Written by`setup`: roster slots, scoring, draft order, your picks.                                     |
| `data/players_nfl.json`                          | Sleeper player table, cached 24h.                                                                        |
| `out/card_latest.md`                             | Current card. Re-written on every board change.                                                          |
| `out/state.json`                                 | Current state + full candidate table (for a second screen).                                              |
| `out/log.jsonl`                                  | Every pick, every card, every candidate set at your turns.                                               |
| `out/packet_<pick>.json`                         | Exactly what the LLM saw for each of your picks.                                                         |
| `out/teamstate_<pick>.md`                        | Post-pick team-state analysis after each of your picks.                                                  |
| `out/script.csv`                                 | Pre-draft survival at each of your picks.                                                                |

## Caviots / Known Gaps

- Sleeper's API exposes picks, draft settings, order, and `last_picked`; it does not say who is on autodraft or expose the clock directly, so both are inferred.
- ADP is a model of the average league, not yours. If your league is RB-heavy or reaches for homers, calibrate `sigma_slope` up (more chaos) and `run_shift` up (they chase runs); In these cases `surv` for the top 2–3 candidates is what matters.
- The `last_picked` timestamp should be checked once on a live draft - if it lags, the poller's own observation times are used and clock display is approximate.
- `search_rank` from the player table stands in for Sleeper's autodraft ranking; if you capture the draft room's own rankings, add a `sleeper_rank` column to `players.csv` and it overrides `search_rank`.
- Players drafted who aren't in your research table are tracked as taken but contribute nothing to opponent-need estimates - this is now the normal path for DEF (still unmodeled by design; K is a modeled, VORP'd position, see **Kickers** above) and for anything else missing from `players.csv`.
- The mock's opponents draft from the same ADP model the survival sim uses.
- Mock-only settings for exercising the harness offline: `mock.autodraft_slots` forces which slots autodraft (overrides the random `autodraft_teams` count); `mock.fault_at_pick` / `mock.fault_n_polls` inject `ConnectionError` for N consecutive polls starting at a given pick, to test the reconnect ladder; `llm.backend: fake` with `llm.fake_delay_s` returns a canned card after a delay, for exercising the LLM queue without burning real calls.
