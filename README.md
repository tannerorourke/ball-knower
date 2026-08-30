# draftkit

A Sleeper Sports snake-draft assistant and agentic research workflow. It researches, follows, and suggests, you make the pick.

An agent workflow shards research on a *per-team* or *per-division* basis of all players in the top-140 ADP. Each agent researches and analyzes recent player news, injuries, opinions, and history, team staffing developments, as well as per-team personnel (committee backfields, shared targets, pass vs. run-heavy schemes) to produce research tables that combine base stats (VORP, bye weeks, etc) with computed metrics.

At draft time, the system uses a deterministic scoring algorithm based on VORP and draft-related metrics and Monte Carlo simulation of draft outcomes from the *current draft state* to actively re-rank remaining players and present suggestions. At pick time (and three picks before, optionally), the system combines the research and derived metrics of the top remaining players with a lightweight model call to provide final model adjudication cards in a A/B/C format.

---

## Setup

**Before draft**:

1. `cd .../draftkit` $\rightarrow$ `pip install -r requirements.txt` $\rightarrow$ set `league_id` (`sleeper.com/leagues/<this_number>/league`) and username in `config.yaml`.
3. `python -m draftkit setup` - confirms roster/scoring parse, seeds `players_skeleton.csv`, and prints your picks + a settings block.
4. (Optional) `python tools/make_fake_research.py data` and `python -m draftkit mock --offline --speed 60` (add --no-llm to skip model card) - dry run so the screen and CLI output is familiar. 

**Draft day (2+ hrs out)**:

1. Research options:
- (best) `python -m draftkit research --by team` - Seed players and base stats (bye week, VORP, etc) from Sleeper's projections API base stats. Research agents analyze team and player news *per-team* (4 concurrent, ~35m runtime) to add judgement data (floor/ceil, risk, role, tags, tiers, po_sched), and emit `players.csv`/`stacks.csv`/`anti.csv`. Booyah!
- (better) `python -m draftkit research` - Seeds base stats, run research agents *per NFL division* (4 concurrent, ~10m) to add judgement data and emit research tables. Best if short on time.
- (ok) `python -m draftkit fetch` - only generate `data/players_skeleton.csv`. Run `RESEARCH.md` in a LM of your choice with the skeleton attached. Place output under `data\`.
2. Once the draft order posts: `python -m draftkit script` - your round-by-round expected board.
3. Edit strategy_notes in `config.yaml` for additional rules you want enforced during your draft.

**Draft Time**:

`python -m draftkit run`. Leave it running while your draft goes.

When you're on the clock the header turns red. The harness's card is already there, the model output card lands 20–40 s later. Once you pick in Sleeper, the next tick sees your pick and moves on.

## Requirements and Configuration

Requires Python 3.10+.

See `config.yaml`.

| key | default | effect |
|---|---|---|
| `poll_interval` | 2.0 | seconds between polls; 2s is ~60 requests/min against a 1000/min limit |
| `top_n` | 8 | candidates shown and sent to the LLM |
| `scoring.replacement_extra` | QB3 RB2 WR2 TE2 | how far past the last starter replacement sits |
| `scoring.ceiling_blend` | {1:0, 6:.15, 9:.3, 11:.5} | round → weight on (ceil − proj) |
| `scoring.need_weight` | 6 | starter bonus |
| `scoring.bench_penalty` | QB12 TE9 RB2 WR2 | per bench body at that position, ×depth |
| `scoring.bye_cost_per_collision` | 4 | |
| `scoring.stack_max_bonus` | 6 | |
| `mc.sims` | 2000 | ~50 ms per recompute |
| `mc.sigma_base/slope` | 1.0 / 0.12 | ADP noise |
| `mc.run_shift` | 0.4 | run momentum, σ units |
| `mc.autodraft_latency_s` | 4 | |
| `llm.backend` | claude-cli | `anthropic` (needs `ANTHROPIC_API_KEY`), `claude-cli`, `none` |
| `llm.model` | "" | model id; empty = backend default |

## Commands

All commands read `config.yaml` (`-c other.yaml` to override).

```bash
python -m draftkit setup                 # pull league + draft settings, cache players, print my picks and a
                                         # settings block to paste into RESEARCH_PROMPT.md.
python -m draftkit script                # Monte Carlo from an empty board: who is likely there at each of
                                         # my picks. --slot N to try a slot before the order is posted.
python -m draftkit run                   # draft day. Dashboard in the terminal; --headless writes files only.
python -m draftkit mock                  # dry run against a simulated draft. --speed 60 finishes in ~3 min,
                                         # --no-llm skips the card call, --slot N, --seed N.
```

---

## Research Tables

`players.csv`: One row per player with projection, floor/ceiling, tier, ADP, risk, role, injury, bye, and tags from a closed vocabulary. 
`stacks.csv` lists each QB with the pass-catchers whose weekly scores correlate with his, scored 0–100. 
`anti.csv` lists same-team pairs that cannibalize each other with a penalty in season points. The full schema and vocabulary is in `RESEARCH_PROMPT.md`; Every table is keyed by `name`

## Metrics

For the 2026 season, I'm drafting and will be testing the results of the below VORP metric against a range of other metrics.

For weight parameters $w_{wait}$ and $w_{vorp}$, cost of waiting $C_{wait}$, ...:

$$
adj = wait\_weight \,\cdot\, max(wait\_cost, 0) + vorp\_weight \,\cdot\, vorp\_eff + need + stack − bye\_cost − anti\_pens − risk\_pen
$$

$$
adj = w_{wait} \,\cdot\, max(C_{wait}, 0) + w_{vorp} \,\cdot\, vorp + need + stack − C_{bye_wk} − anti − risk\_pen
$$

Within a position, this orders by VORP; across positions it orders by what waiting for the player costs, with roster fit as points-denominated adjustments. In terms of drafting, the live LLM reorders only based on remaining players and reasons stated in these numbers.

**proj / floor / ceil**: Season points in your league's scoring. Floor $\approx$ 20th percentile outcome, ceil ≈ 80th. In rounds 1–5 the scorer values `proj`; from round 6 it blends toward `ceil` (`scoring.ceiling_blend`), so late picks are judged on upside.

**tier / cliff**: Tiers come from the research (gap-based, per position). Cliff is the points lost if you miss a tier and get the top of the next one. The dashboard shows how many players are left in each tier; the MC reports the probability each tier is empty by your next pick (`tier_empty_p`).

**Replacement level**: Computed from your league's roster, not a rule of thumb. All teams' fixed starting slots are filled greedily by projection, then the flex slots are filled from the best remaining RB/WR/TE, and replacement is the best player left at each position plus `replacement_extra` more (QB and TE get a bigger shift because they're streamable). With two W/R/T flexes this lands around RB30 / WR38 / TE13 / QB15 - pull the exact numbers from `validate`.

**VORP**: `value − replacement[pos]`. This is the comparison scale across positions. Two adjustments: (1) a player who would sit in a flex or bench slot is measured against the *flex* replacement (the highest of the RB/WR/TE replacement levels), because that's who he actually competes with — this is what stops a TE2 from looking like a top-40 pick; (2) from round 6, `value` includes the ceiling blend above.

**ADP**: Overall pick number from Sleeper's ADP. Drives the survival model. `adp_delta` (your rank − ADP) is not shown directly; the survival model is a better version of the same information.

**surv %**: The probability this player is still available at your *next* pick, from the Monte Carlo. When you're on the clock it answers "if I don't take him now, will I get another chance?" - a player at 85% can wait, a player at 5% can't. At the turn (slots 1 and 12) the horizon is your next non-consecutive pick.

**E[best at next] and wait_cost**: For each position, the Monte Carlo estimator reports the expected best VORP still available at your next pick. `wait_cost = vorp − E[best at next][pos]` is what waiting one turn costs you at that position. This is the actual draft-timing criterion: if you'll fill both positions eventually, drafting position P now over Q is right iff `wait_cost(P) > wait_cost(Q)`. A deep position has low wait cost regardless of how good its top player is. `wait_cost` is scaled down for slots you don't need to fill (×0.8 flex, ×0.3 bench, ×0 capped).

**need**:: Where this player would land on your roster right now: 
- `STARTER` (open fixed slot at their position, + `need_weight`)
- `FLEX` (open flex, +half)
- `BENCH` (a penalty from `bench_penalty`, large for QB/TE since backups there rarely start, halved from `bench_free_round`).
- `CAPPED` (you've hit `pos_caps`). Starter bonuses double when rounds left ≤ open starters + 1.

**bye_cost**: Points charged when the candidate shares a bye with players already holding his position's fixed slots, i.e. both RB slots dark in the same week. `bye_cost_per_collision` (default 4) is roughly one week of starter-minus-replacement. It is deliberately smaller than most tier gaps: a bye collision should break a tie, not skip a tier.

**stack**: `stack_max_bonus × stack_score/100` when you roster the other half of a `stacks.csv` pair. Same logic: tiebreaker inside a tier, not a reason to reach.

**anti**: Sum of `anti.csv` penalties against players you already roster (committee backfields, shared targets, rushing QB with his own RB).

**risk**: 0–5 from the research rubric, charged at `risk_weight` points per level in rounds 1–6 and half that afterwards.

## Monte Carlo Survival Model

Every tick, for the opponent picks between now and your horizon pick, the system runs Monte Carlo for 2,000 simulated drafts. Each human team draws a personal board `adp + σ(adp)·ε` with `σ = sigma_base + sigma_slope·adp` (ADP standard deviation, ~2 picks at pick 10, ~7 at pick 50, ~13 at pick 100), shifted in $\sigma$ units by roster need (`need_shift`: a position they're capped at is pushed 20σ down, an open starter is 0, bench-only is 1.5σ, urgent open starters late pull up) and by run momentum (`run_shift` $\sigma$ per same-position pick in the last `run_window` picks). They take the lowest score. Autodraft teams are deterministic and take Sleeper's top-ranked available player that fits their roster - a run of only autodrafters before you narrows the tree.

A team is classified autodraft when the median gap between the previous pick and theirs is under `autodraft_latency_s` across ≥2 picks. Several picks landing in one poll count as instant. You can force slots in `mc.autodraft_slots` if you know who's on autopilot.

## Output card

```
A  J. Jacobs   RB  T3  vorp 41.0  surv 18%  bye 7 ok   stack —    need STARTER
   + waiting costs ~14 pts at RB (tier closes p=82%, 1 left, cliff 14)
   - risk 3, SHARED
B  ...
IF A gone -> B (...)
IF A,B gone -> C (...)
```

The harness writes this instantly on every board change. When you're on the clock the same packet (`out/packet_<pick>.json`: candidates with every number above, your roster, open slots, tiers remaining, what each opponent before your next pick still needs, `strategy_notes`) goes to the LLM, which returns the same shape with its own +/− lines and IF logic. Guardrails: the candidate set is closed (top `top_n` by `adj`), the model is told it may not compute or recall, and its output is parsed — any option naming a player outside the set is rejected and the harness card stands, with the reason in the log. Set `llm.at_n_minus_2: true` for an additional card two picks out (one more call per turn; the on-clock card is the one that has the exact board).

`strategy_notes` in `config.yaml` is free text the LLM sees. Keep it to rules you actually want enforced ("no TE2 before round 12", "stack only inside a tier").

## System Files

| path | what |
|---|---|
| `RESEARCH_PROMPT.md` | The prompt you run 1–2 hours before the draft. Produces the three CSVs. |
| `config.yaml` | League ids, model knobs, LLM backend, strategy notes. |
| `data/players.csv`, `stacks.csv`, `anti.csv` | Research output. Put them here. |
| `data/league.json` | Written by `setup`: roster slots, scoring, draft order, your picks. |
| `data/players_nfl.json` | Sleeper player table, cached 24h. |
| `out/card_latest.md` | Current card. Re-written on every board change. |
| `out/state.json` | Current state + full candidate table (for a second screen). |
| `out/log.jsonl` | Every pick, every card, every candidate set at your turns. |
| `out/packet_<pick>.json` | Exactly what the LLM saw for each of your picks. |
| `out/script.csv` | Pre-draft survival at each of your picks. |
| `tools/make_fake_research.py` | Synthetic research for dry runs: `python tools/make_fake_research.py data` (overwrites `data/*.csv`). |

## Caviots / Known Gaps

- Sleeper's API exposes picks, draft settings, order, and `last_picked`; it does not say who is on autodraft or expose the clock directly, so both are inferred. 
- ADP is a model of the average league, not yours. If your league is RB-heavy or reaches for homers, calibrate `sigma_slope` up (more chaos) and `run_shift` up (they chase runs); In these cases `surv` for the top 2–3 candidates is what matters.
- The `last_picked` timestamp should be checked once on a live draft — if it lags, the poller's own observation times are used and clock display is approximate. 
- `search_rank` from the player table stands in for Sleeper's autodraft ranking; if you capture the draft room's own rankings, add a `sleeper_rank` column to `players.csv` and it overrides `search_rank`. 
- Players drafted who aren't in your research table are tracked as taken but contribute nothing to opponent-need estimates. K and DEF aren't modeled. 
- The mock's opponents draft from the same ADP model the survival sim uses.
