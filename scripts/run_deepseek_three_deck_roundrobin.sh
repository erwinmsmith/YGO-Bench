#!/usr/bin/env bash
# Run a six-game, directional round robin for a single DeepSeek self-play policy.
# The two agent identifiers create independent seat sessions; they do not denote
# different underlying models.
set -u

cd /root/rivermind-data/YGO
set -a
. ./.env
set +a

PYTHON_BIN=/root/miniconda3/envs/YGO/bin/python
RUN_ID=deepseek_three_deck_roundrobin_001
AGENT_P0=react-fast-seat0:deepseek:deepseek-v4-flash
AGENT_P1=react-fast-seat1:deepseek:deepseek-v4-flash

run_duel() {
  local deck0="$1"
  local deck1="$2"
  local seed="$3"

  printf '\n=== %s: P0=%s P1=%s seed=%s ===\n' "$RUN_ID" "$deck0" "$deck1" "$seed"
  "$PYTHON_BIN" -m ygobench.experiments.cli duel \
    --run-id "$RUN_ID" \
    --deck1 "$deck0" \
    --deck2 "$deck1" \
    --agent1 "$AGENT_P0" \
    --agent2 "$AGENT_P1" \
    --seed "$seed" \
    --max-decisions 0
}

# Each directed matchup gets an independent engine seed.  This avoids repeating
# the same seat/deck shuffle while preserving an auditable deterministic setup.
run_duel BlueEyes TenyiSword 20260821
run_duel TenyiSword BlueEyes 20260822
run_duel BlueEyes CyberDragon 20260823
run_duel CyberDragon BlueEyes 20260824
run_duel TenyiSword CyberDragon 20260825
run_duel CyberDragon TenyiSword 20260826
