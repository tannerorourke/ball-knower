# Configuration reference

Long-form descriptions for all fields in `config.yaml`. Values here override code defaults in `cli.py DEFAULTS`.

## League / Sleeper

| key | default | role |
|---|---|---|
| `league_id` | `""` | Your Sleeper league, from the URL `sleeper.com/leagues/<league_id>`. Used by `setup` to pull roster shape, scoring settings, and the draft. |
| `draft_id` | `""` | Pin a specific draft. Empty = the league's current draft. |
| `username` | `""` | Your Sleeper username; resolved to a draft slot once the draft order is posted. |
| `my_slot` | `null` | Manual override of your draft slot (1-based) if username lookup fails or the order isn't posted yet. |

## Files & cadence

| key | default | role |
|---|---|---|
| `data_dir` | `data` | Inputs: `league.json`, `players_skeleton.csv`, the research CSVs, the cached Sleeper player table. The Sleeper client also derives which positions to fetch from `league.json` here (falls back to all modeled positions before `setup` has run). |
| `out_dir` | `out` | Outputs: `card_latest.md`, `packet_<pick>.json`, `teamstate_<pick>.md`, `state.json`, `log.jsonl`, `my_roster.csv`. |
| `poll_interval` | `2.0` | Seconds between polls to the Sleeper client to update draft state. Also sets the batched-pick latency estimate (half an interval). |
| `top_n` | `12` | Candidates ranked in the dashboard table and sent to the live-draft LLM. |
| `seed` | `0` | RNG seed for the Monte Carlo. 0 = random each run; set it to make sims reproducible. |

## Monte Carlo survival model (`mc.*`)

Simulation of the opponent picks between now and your next pick to answer "who is still on the board when it's my turn?". The `surv`, `wait_cost` (via expected best VORP at your next pick), and `tier_empty_p` are derived from this.

| key | default | role |
|---|---|---|
| `mc.sims` | `2500` | Simulated drafts per recompute (2500 $\approx$50ms; raise for smoother survival numbers, lower for speed. |
| `mc.pool` | `80` | Top-N *available* players by ADP treated as pickable in the sim (relative - slides down the board as players go). Auto-raised to cover the pick window + 40. Players outside the pool are assumed 100% safe. |
| `mc.sigma_base` / `mc.sigma_slope` | `1.0` / `0.12` | ADP standard deviation: `σ = base + slope·adp` (~2 picks of noise at ADP 10, ~13 at ADP 100). All shifts below are expressed in these σ units, so they mean the same thing early and late. |
| `mc.run_shift` | `0.4` | Run momentum: each same-position pick in the last `run_window` picks pulls that position up every simulated board by this many σ — models positional runs feeding themselves. |
| `mc.run_window` | `4` | How many recent picks count toward run momentum. |
| `mc.need_shift` | `STARTER:0 FLEX:0.5 BACKUP:1.0 DEPTH:1.5 URGENT:1.0 CAPPED:20` | Per-roster-state shift (σ) applied to each simulated opponent's board, from *their* actual roster: an open starter is neutral (URGENT pulls it up late), depth-only positions drift down, capped positions are effectively off their board. |
| `mc.autodraft_latency_s` | `3.0` | A slot is classified autodraft when `min(median, mean)` of its observed time-on-clock is below this (needs ≥3 picks). Autodrafters are simulated deterministically: Sleeper's top-ranked available player that fits their roster. |
| `mc.autodraft_slots` | `[]` | Force slots to be treated as autodrafting, e.g. `[3, 9]`, when you know who's on autopilot. |
| `mc.pos_caps` | `QB:2 TE:2 RB:7 WR:7 K:1` | Roster caps assumed for *opponents* in the sim (an opponent at the cap won't take another). Note this is separate from `scoring.pos_caps`, which caps *you*. |
| `mc.slot_tendencies` | `true` | Learn each human slot's style live from their own picks: a reach-noise multiplier (rms of `adp − pick_no` vs what σ predicts — reachers get wider boards, chalk drafters tighter) and a per-position lean vs the room (σ shift). Shown in the LLM packet as `style: noise x1.3, leans WR`. |
| `mc.tendency_prior_picks` | `4` | Shrinkage for the learned style: estimates blend toward neutral until a slot has more than this many picks, so one wild pick doesn't rewrite a drafter's profile. |
| `mc.lean_shift` | `1.0` | Scale of the positional-lean shift: σ per 100% lean vs the room's position share. |

## Deterministic Candidate Ranking (`scoring.*`)

Produces `adj = wait_w[tag]·max(wait_cost,0) + vorp_weight·vorp_eff + need + stack − anti − risk_pen`.
- Bye weeks are deliberately **not** in `adj` — they surface as warnings (`bye_team_count`) and shade the BACKUP bonus instead. 
- The need machinery is a state machine: fill starters, then direct backups, then maximize.

| key | default | role |
|---|---|---|
| `scoring.ceiling_blend` | `{1:0, 6:0.15, 9:0.30, 11:0.50}` | Round → weight on `(ceil − proj)`. Early rounds draft the projection; late rounds increasingly draft the ceiling (upside hunting). |
| `scoring.need_weight` | `6.0` | Points for filling an open starting slot — the base unit every other need bonus/penalty scales from. |
| `scoring.backup_weight` | `0.8` | BACKUP bonus = `backup_weight × need_weight`. "Less-so but still equally important" than a starter. |
| `scoring.backup_targets` | `QB:1 RB:2 WR:2 TE:1 K:0` | Direct backups to roster per position; beyond this a pick is DEPTH. Only activates once every non-K starter is filled. |
| `scoring.backup_bye_overlap_factor` | `0.4` | Multiplier on the BACKUP bonus when the candidate shares a bye with your same-position players — a backup only *covers* the starter if their byes differ. |
| `scoring.depth_while_starters_open` | `15.0` | Flat penalty on any DEPTH pick while a non-K starter slot is open — the round-11 "6th WR over an open RB2" gate. Kickers never trigger this gate. |
| `scoring.depth_while_backups_open` | `6.0` | Softer gate: DEPTH pick penalty while backup targets are unfilled. |
| `scoring.wait_w` | `STARTER:1.0 FLEX:0.8 BACKUP:0.6 DEPTH:0.15 CAPPED:0 K-WAIT:0` | Per-need-tag weight on `wait_cost` — waiting only costs something if you actually intend to fill the slot; DEPTH's appetite is nearly off. |
| `scoring.capped_multiplier` | `3.0` | CAPPED penalty = `−capped_multiplier × need_weight` once you hit `scoring.pos_caps`. |
| `scoring.urgency_factor` | `2.0` | Multiplier on STARTER/FLEX/BACKUP bonuses once rounds left ≤ open slots + 1 (a DEF slot eats a round from this math). Also opens the kicker's endgame window. |
| `scoring.risk_weight` | `1.0` | Points subtracted per research risk level (0–5); halved after round 6, when upside swings matter more than safety. |
| `scoring.replacement_extra` | `QB:3 RB:2 WR:2 TE:2 K:1` | Replacement level = the last league-wide starter *plus* this many players — benches soak up a few starters' worth, and QB/TE are streamable. |
| `scoring.vorp_weight` | `0.15` | Weight on the candidate's (flex-rebased) VORP inside `adj` — the tiebreaker among players passing the same need gate. |
| `scoring.bench_penalty` | `QB:12 TE:9 RB:2 WR:2 K:15` | Per-position penalty per bench body for a DEPTH pick (×depth). Onesies punish hard — a QB3 rarely starts; RB/WR depth is cheap injury insurance. |
| `scoring.bench_free_round` | `9` | Round from which DEPTH penalties halve — late bench picks are fine. |
| `scoring.pos_caps` | `QB:3 TE:2 RB:7 WR:7 K:1` | Your own hard caps; at the cap a position goes CAPPED (heavy penalty, zero wait weight). |
| `scoring.stack_max_bonus` | `6.0` | Points at `stack_score` 100 for correlating with your QB/receivers (scales linearly with the researched score). |
| `scoring.bye_cost_per_collision` | *(removed)* | Bye cost no longer enters `adj`; kept only for the display column. |

## In-season management (`inseason.*`)

Weekly projections are Sleeper's per-week stat lines scored with the league's scoring settings. A player whose team is on bye that week scores 0. A game-status tag (Questionable, Doubtful, Out) describes one game, so it adjusts only the current NFL week: Out scores 0 there, and in every other week the raw projection stands. IR, PUP, suspended, COVID list, NA and DNR score 0 in every week. A player's `ros` is his projection summed from the report week through week 17 under the same rules.

`lineup`, `waivers` and `manage` take my roster as it stands after the reserve moves `injuries` recommends: a bench player eligible for a free reserve slot counts on reserve, and a reserve player with no tag counts back in play. The reports say so when a move applies. Among drops tied on gain, the recommended one has the lowest `ros`, and `drop_order` in the manage packet lists the players outside the report week's best lineup (reserve and taxi aside) by `ros`, lowest first. The best lineup decides, since the starters set in Sleeper may be last week's. The waiver candidates are the projection pool below plus every trending free agent the player table holds at a position the league starts, so a trending pickup ranks by gain beside the rest.

The first two keys below scale the projection for players still expected to be limited; the next two size the waiver search; the last two size the `usage` trend window and its drop flag.

| key | default | role |
|---|---|---|
| `inseason.questionable_discount` | `0.85` | Multiplier on the current-week projection of a player tagged Questionable. |
| `inseason.doubtful_discount` | `0.4` | Multiplier on the current-week projection of a player tagged Doubtful. |
| `inseason.waiver_pool` | `40` | `waivers` scores the top this-many free agents (players on no roster) by projection over the horizon, plus the trending free agents at started positions. |
| `inseason.horizon_weeks` | `3` | `waivers` sums the optimal-lineup gain of each add/drop pair over this many weeks starting at `--week`; a pickup that covers a bye week scores for it. |
| `inseason.usage_weeks` | `3` | `usage` looks at this many of the most recent completed weeks before `--week`; weeks with no stat rows are skipped. |
| `inseason.usage_drop` | `0.15` | `usage` flags a rostered player whose snap, target or carry share fell by this much or more from the first week in the window to the last. |

## Weekly report (`python -m sleeper_agent manage`)

`manage` writes `out/manage/w<N>/report.md` and `packet.json`. Unless `--no-llm` is passed or `llm.backend` is `none`, it makes one model call in the `manage` phase (`llm.manage_model`, `llm.manage_timeout_s`, `llm.manage_effort`) and keeps the reply only if it passes validation; otherwise the engine's report stands with the reason in its header, and a rejected reply goes to `rejected.md`. On the `anthropic` backend a `manage` call may return up to 4000 tokens and a `draft` call 1200. It recommends and never writes to Sleeper.

| flag | role |
|---|---|
| `--week N` | Report week. Default: the current NFL week. |
| `--as-of-week N` | Replay week N: rosters and FAAB are rebuilt to 12:00 UTC the day after week N-1's last game by undoing later transactions, and the report goes to `out/manage/w<N>-replay/`. Past injury statuses, lineups and reserve lists cannot be recovered; every player counts as healthy, and no research set or trending count is read. |
| `--note TEXT` | Free text added to `notes` in the packet, which the model reads. Repeatable. |
| `--watch NAME` | A player to include; a free agent joins the adds whatever its gain. Repeatable. |
| `--no-llm` | Skip the model call. |
| `--model M` | Override `llm.manage_model` for this run. |

## In-season research (`research.*`)

`research --in-season` writes one refresh set per week under `data/research/<season>-w<N>/` and names the latest in `data/research/current.json`. The prompts and replies live in `out/research/<season>-w<N>/`. It uses the `llm.*` backend, and `inseason.waiver_pool` and `inseason.horizon_weeks` to pick candidates.

`--teams` picks the shards, one per NFL team that has rows. `auto` (default) takes as candidates the adds of the 8 best add groups by gain (the `waivers` ranking) plus the 5 most-added trending free agents at started positions, and shards the teams of those candidates and of my players. `ALL` or a comma list such as `ARI,BUF` takes the whole `inseason.waiver_pool` plus every trending free agent at a started position as candidates; a list then keeps only its teams, and a listed team with no rows is skipped with a note. A reply row for a player outside its shard's slice is dropped. Rows of the previous set stay, with their original `as_of`, for a player the new replies hold no row for in that table (`players.csv` and `news.csv` separately).

| key | default | role |
|---|---|---|
| `research.max_age_days` | `7` | `load_current` returns a warning when the latest refresh set is older than this many days, or when none exists. |

## Model card and Backend (`llm.*`)

All calls run through one FIFO worker queue: jobs are never cancelled, the panel keeps the
previous card until a finished job replaces it, and post-pick analysis always completes
before the next card fires.

| key | default | role |
|---|---|---|
| `llm.backend` | `none` (yaml: `claude-cli`) | Shared by research + draft. `claude-cli` = your Claude subscription via the CLI; `anthropic` = API billing (`ANTHROPIC_API_KEY`); `openai-compat` = any OpenAI-schema endpoint; `none` = deterministic card only; `fake` = canned card after `fake_delay_s`, for testing the queue. |
| `llm.research_model` | `""` | Model for the pre-draft judgment pass (accuracy matters, latency doesn't). Empty = backend default. |
| `llm.research_effort` | `high` | Thinking depth for research shards (`low`–`max`, claude-cli backend). |
| `llm.draft_model` | `""` | Model for live cards and post-pick analysis (latency matters — you're on a clock). |
| `llm.draft_effort` | `""` (yaml: `medium`) | Thinking depth for live cards and post-pick analysis. Empty inherits the backend default — for claude-cli, whatever the user-level session saved, which can be slow. Keep `low`–`medium`: past that, calls outlast the pick clock. |
| `llm.base_url` | `""` | openai-compat only: the endpoint, e.g. `https://openrouter.ai/api/v1` or a local `http://localhost:11434/v1`. |
| `llm.api_key_env` | `LLM_API_KEY` | Name of the env var holding the openai-compat key; unset = no auth header (local models). |
| `llm.research_timeout_s` | `900` | Per-shard timeout for the research fan-out (`--timeout` flag overrides). |
| `llm.draft_timeout_s` | `90` | Timeout for a live card call — past this the deterministic card stands alone. |
| `llm.manage_model` | `""` | Model for the `manage` report call (`--model` overrides). Empty = backend default. |
| `llm.manage_timeout_s` | `180` | Timeout for the `manage` report call; past it the engine's report stands. |
| `llm.manage_effort` | `""` | Thinking depth for the `manage` report call. Empty inherits the backend default. |
| `llm.at_n_minus` | `false` (yaml: `true`) | Pre-generate the card before you're on the clock (one extra call per turn) so it's ready when the clock starts. |
| `llm.n_minus` | `2` | Count rule: fire the early card when ≤ this many picks remain before yours. |
| `llm.eta_trigger_s` | `60` | Time rule (from round 4): also fire when the expected wait to your pick — sum of `min(median, mean)` pick time over the slots ahead — drops to this. A wall of autodrafters ahead fires the card *more* picks out than `n_minus`. |
| `llm.post_pick` | `true` | After your pick, run a team-state analysis (on/off plan vs strategy notes, top holes, bye pile-ups). At the snake wheel the back-to-back pair gets one analysis, after the second pick. |
| `llm.fake_delay_s` | `2.0` | `fake` backend only: artificial latency, for exercising queue ordering in mocks. |

## `strategy_notes`

Free text passed verbatim to the card model and the post-pick analysis — your draft philosophy in prose (round ranges, positional rules, stack policy). This is the paramount lever for steering the SLEEPER pick; the deterministic scorer never reads it.

## Offline dry runs (`mock.*`, `python -m sleeper_agent mock`)

| key | default | role |
|---|---|---|
| `mock.teams` / `mock.rounds` | `12` / `14` | Simulated league shape. |
| `mock.my_slot` | `10` | Your slot in the fake draft (`--slot` overrides). Use `1` or `12` to exercise the wheel. |
| `mock.pick_timer` | `180` | Simulated clock, seconds. |
| `mock.roster_positions` | QB 2RB 2WR TE 2FLEX 6BN IR | Roster shape of the fake league (add `K`/`DEF` to test those paths). |
| `mock.autodraft_teams` | `3` | How many random slots autodraft. |
| `mock.autodraft_slots` | `[]` | Pin *which* slots autodraft (overrides the random count) — makes timing scenarios reproducible. |
| `mock.human_min_s` / `mock.human_max_s` | `8` / `45` | Range of the fake humans' per-pick latency. |
| `mock.my_pick_delay_s` | `20` | How long the mock waits before submitting your queued pick. |
| `mock.speed` | `10` | Wall-clock divisor — 10 makes a 60s pick take 6s. The autodraft threshold and `eta_trigger_s` are divided by this automatically so detectors stay calibrated. |
| `mock.fault_at_pick` | `0` | Pick count at which the mock client starts raising `ConnectionError` (0 = never) — tests the reconnect ladder. |
| `mock.fault_n_polls` | `0` | How many consecutive polls fail once triggered, then normal service resumes. |
