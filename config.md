# Config

Long-form descriptions for configurable fields in `config.yaml`.

| key | description | effect |
|---|---|---|
| league_id |  |  |
| username |  |  |
| draft_id |  |  |
| data_dir |  |  |
| out_dir |  |  |
| poll_interval |  |  |
| top_n |  |  |
| seed |  |  |
| strategy_notes |  |  |

## mc (Monte Carlo Simulation)

| key | description | effect |
|---|---|---|
| sims |  |  |
| pool |  |  |
| sigma_base |  |  |
| sigma_shift |  |  |
| run_shift |  |  |
| run_window |  |  |
| need_shift |  |  |
| autodraft_latency |  |  |
| autodraft_slots |  |  |
| pos_caps |  |  |

## scoring

| key | description | effect |
|---|---|---|
| ceiling_blend |  |  |
| need_weight |  |  |
| backup_weight |  |  |
| backup_targets |  |  |
| depth_while_starters_open |  |  |
| depth_while_backups_open |  |  |
| backup_bye_overlap_factor |  |  |
| wait_w |  |  |
| capped_multiplier |  |  |
| urgency_factor |  |  |
| stack_max_bonus |  |  |
| risk_weight |  |  |
| replacement_extra |  |  |
| vorp_weight |  |  |
| bench_penalty |  |  |
| bench_free_round |  |  |
| pos_caps |  |  |

## llm

| key | description | effect |
|---|---|---|
| backend |  |  |
| research_model |  |  |
| draft_model |  |  |
| base_url |  |  |
| api_key_env |  |  |
| research_timeout_s |  |  |
| at_n_minus |  |  |
| n_minus |  |  |
| eta_trigger_s |  |  |
| post_pick |  |  |

## mock

| key | description | effect |
|---|---|---|
| teams |  |  |
| rounds |  |  |
| my_slot |  |  |
| pick_timer |  |  |
| roster_positions |  |  |
| autodraft_teams |  |  |
| autodraft_slots |  |  |
| human_min_s |  |  |
| human_max_s |  |  |
| my_pick_delay_s |  |  |
| speed |  |  |
| fault_at_pick |  |  |
| fault_n_polls |  |  |

