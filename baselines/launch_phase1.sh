#!/bin/bash
# Phase 1 (healthy) for the phase1->phase2 warm-start pipeline. Trained with the SAME
# teacher-MLP architecture as Phase 2 (no injury, no pain) so the warm-start into
# Phase 2 is dimension/architecture compatible. gait_tuning=1 yields a clean ~2.6 Hz
# physiological walk (not high-frequency bounding).
#
# Self-contained / portable: paths relative to this repo, no systemd, no hardcoded
# conda. Activate your Isaac Lab python env BEFORE running.
#
# Usage: ./launch_phase1.sh [SEED]        (default seed 42)
# Output: <repo>/scripts/rsl_rl/logs/rsl_rl/unitree_go1_rough_teacher/*phase1_mlp_s<SEED>/
set -uo pipefail
S="${1:-42}"
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS="$(cd "$BASE_DIR/.." && pwd)/scripts/rsl_rl"
echo "[phase1] healthy teacher-MLP, seed=$S  scripts=$SCRIPTS"
cd "$SCRIPTS"

GO1_NO_WARMSTART=1 GO1_INJURY_ONEHOT=1 GO1_PROPRIO_ONLY=1 GO1_FLAT_TERRAIN=1 GO1_PHASE2_GAIT_TUNING=1 \
GO1_PROB_PEG_LEG=0.0 GO1_USE_PEG_LEG_CURRICULUM=0 GO1_PAIN_WEIGHT=0.0 \
GO1_INJURED_FORCE_NONUSE_WEIGHT=0.0 GO1_INJURED_DUTY_NONUSE_WEIGHT=0.0 \
GO1_PD_ACTUATOR=1 GO1_PD_KP=20.0 GO1_PD_KD=0.5 \
GO1_STRICT_TERMINATIONS=1 GO1_BAD_ORIENTATION_LIMIT=0.8 GO1_DOMAIN_RAND=1 \
GO1_CMD_VX_MIN=0.10 GO1_CMD_VX_MAX=0.30 GO1_CMD_VY_ABS=0.0 GO1_CMD_YAW_ABS=0.15 \
GO1_FLAT_ORIENTATION_WEIGHT=-2.0 GO1_SURVIVAL_BONUS_WEIGHT=1.0 \
GO1_CONTACT_FORCE_ASYM_WEIGHT=0.0 GO1_DUTY_FACTOR_ASYM_WEIGHT=0.0 GO1_DIAGONAL_LOAD_ASYM_WEIGHT=0.0 \
GO1_FRONT_REAR_LOAD_DIST_WEIGHT=0.0 GO1_TROT_SYNC_WEIGHT=0.0 \
PHASE2_RUN_NAME="phase1_mlp_s${S}" PHASE2_MAX_ITER="${PHASE2_MAX_ITER:-6000}" \
NUM_ENVS="${NUM_ENVS:-2048}" SEED="$S" \
bash ./train_phase2.sh
