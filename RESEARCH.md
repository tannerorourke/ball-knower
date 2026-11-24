# Draft Research Prompt

For Claude/User: This is the manual, single-conversation alternative to `python -m ball_knower research`. It consumes `data/players_skeleton.csv` - Sleeper's own projections, ADP, and schedule: name, pos, team, bye, proj (scored in your league's settings), tier, adp, inj, age. This prompt performs the judgement on top: Reading team and player news, insights, and projections. Any model works; one with web access does the news sweep, one without still returns a usable board. Prompt starts below the horizontal line in a *new conversation*, and should be run after `python -m ball_knower setup`. 

Before running, check that `data\league.json`, `data\players_nfl.json`, and `data\players_skeleton.csv` exist/are attached.

*Alternative (deep-research pairing, optional)*: If you want a research-mode model (long tool budget, writes reports, bad at emitting strict CSVs) to do the digging: run it first with just the method section above and "produce a news digest for the top 140 by ADP: per player, one line - role, injury, risk factors, stack notes." Then give *any* model the digest + skeleton + this prompt; stage two needs zero tool calls. Two stages beat one whenever the researcher and the formatter want different models.

---

You are the judgment layer for a fantasy football draft assistant. You receive `players_skeleton.csv`, whose numeric columns were filled programmatically from projection data. Your output is consumed by a program: three CSV files and nothing else, every value a concrete number or a token from a closed vocabulary. No commentary, no "N/A", no markdown inside the CSVs.

## League

```
Season: <<2026>>   Platform: Sleeper   Teams: <<12>>
Draft: <<snake>>, <<14>> rounds
Roster: <<QB, RB, RB, WR, WR, TE, FLEX, FLEX, BN x6, IR x1>>   (FLEX = RB/WR/TE. K rows ride along in the skeleton and get judged like any other position when your roster has one - see the risk rubric below. DEF is out of scope: no row, no judgment.)
Scoring: <<paste from `python -m ball_knower setup`>>
```

## What is locked and what is yours

**Locked - copy through unchanged on every row:** `name`, `pos`, `team`, `bye`, `proj`, `adp`. These came from data. Do not rename, re-project, reorder positions, or drop or add rows.

**Yours - overwrite:** `floor`, `ceil`, `risk`, `role`, `inj`, `po_sched`, `tags`, and `tier` (within the rules below). The skeleton holds placeholder defaults (floor/ceil = ±14%, risk 2, role UNKNOWN, po_sched 3); every row you leave at defaults is a row you contributed nothing to.

Work the **top 140 by `adp` thoroughly**; below that, adjust only where you know something concrete and leave the rest at defaults.

## Method

1. **News sweep, last 48 hours,** top 140 by ADP: injuries, holdouts, suspensions, depth-chart and coaching changes, trades. Express findings only through the columns - never prose. Without web access, use what you know and prefer UNKNOWN/default over guessing.
2. **Role.** LOCKED = unambiguous starter with ≥65% of his position's opportunity. SHARED = committee or clear 1B. CONTESTED = job in question. UNKNOWN = no reliable information.
3. **Risk (0–5 int).** 0 = locked role, healthy, prime age. 1 = one minor concern. 2 = moderate injury history OR mild role competition. 3 = active injury designation OR real role competition OR age cliff (RB ≥28, WR ≥31, TE ≥32). 4 = two of those. 5 = holdout, suspension, PUP, or expected to miss ≥3 games. **Kicker rows (pos K):** role LOCKED unless there's an open camp battle; risk 0–2 in almost all cases - don't invent injury/role narratives for kickers, keep floor/ceil tight (kicking volume tracks the offense, not skill variance), and tags rarely apply.
4. **inj.** Correct the skeleton's value where news is fresher: OK, Q, D, O, PUP, HOLDOUT, SUSP.
5. **Floor / ceiling.** Replace the ±14% placeholders: floor ≈ 20th-percentile season outcome, ceil ≈ 80th, floor < proj < ceil always. Widen for injury history, contested roles, rookies, new teams; tighten for locked veteran volume.
6. **Tiers.** The skeleton's tiers are pure projection-gap breaks. You may merge or split them where the gap misleads (e.g. two elite players split by a 7-point gap belong together). Rules: per position, contiguous integers starting at 1, non-increasing in `proj`.
7. **po_sched.** Weeks 15–17 matchups for the player's position: 1 brutal … 5 soft. Projected defense quality, not last season's rank alone.
8. **tags.** 0–3 per row, `|`-separated, from exactly: NEW_OC, NEW_QB, NEW_TEAM, ROOKIE, ROOKIE_THREAT, HANDCUFF, RZ_HOG, YAC, DEEP, VOLUME, CONTRACT_YR, AGE_CLIFF, IR_STASH, INJ_HISTORY, BREAKOUT, DECLINE. `IR_STASH` = currently O/PUP/SUSP with credible return by week 8. `HANDCUFF` = primary backup to a LOCKED RB.

Then build two tables the skeleton cannot know:

9. **stacks.csv** - for each starting QB, every RB/WR/TE on his team with ≥12% projected target share (RBs ≥8%). `stack_score` 0–100 rises with target share, red-zone share, and QB pass volume; deep/high-aDOT profiles score higher on ceiling correlation than YAC/slant profiles. Columns: `qb,rec,team,stack_score,type,tgt_share,rz_share,adot` with type ∈ VOLUME, DEEP, RZ, YAC; shares 0–1 two decimals; adot one decimal. `qb`/`rec` must match `name` values in players.csv exactly.
10. **anti.csv** - same-team pairs that cannibalize each other. Always include: both RBs of a committee (penalty 6–10), top two WRs of a low-volume passing offense (3–5), rushing-heavy RB with his own QB (1–2). Columns: `p1,p2,team,penalty,reason` with reason ∈ COMMITTEE, TARGET_SHARE, RUSH_QB, GOAL_LINE; penalty in season points, one decimal. Names must match players.csv.

## Output

Emit exactly three files: `players.csv` (all skeleton rows, same columns, same order), `stacks.csv`, `anti.csv`. If you are running through a file-system CLI inside a ball-knower folder, save the files to `ball-knower\data`. Process the board in chunks internally if needed, but emit each file once, whole.

## Self-check before you emit (fix silently, do not report)

- Row count and the locked columns are identical to the skeleton.
- floor < proj < ceil on every row; every categorical value in vocabulary, exact case.
- Tiers per position: contiguous from 1, non-increasing in proj.
- Every stacks/anti name resolves to a players.csv `name`.
- No text outside the three files.
