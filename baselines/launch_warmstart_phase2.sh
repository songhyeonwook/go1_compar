#!/bin/bash
# Phase 2 (peg-leg + antalgic pain) WARM-STARTED from the clean Phase 1 (launch_phase1.sh).
# This is the faithful phase1->phase2 curriculum (paper Section 4.6): a clean walking
# policy is loaded, then the functional-splint injury and the nociceptor pain penalty
# (eq. 4) are added on top so an antalgic, partially-loaded gait emerges from a
# physiological baseline instead of a degenerate from-scratch one.
#
# gait_tuning=1 keeps the gait clean (anti-buzz, symmetry-neutral); moderate viability
# floor keeps the injured limb bearing partial load (avoids both non-use and over-use).
#
# Self-contained / portable. Activate your Isaac Lab python env BEFORE running.
#
# Usage: ./launch_warmstart_phase2.sh [SEED] [PHASE1_CKPT]
#   SEED         default 42
#   PHASE1_CKPT  path to the Phase 1 model_*.pt; if omitted, auto-finds the latest
#                phase1_mlp_s<SEED> checkpoint produced by launch_phase1.sh.
set -uo pipefail
S="${1:-42}"
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS="$(cd "$BASE_DIR/.." && pwd)/scripts/rsl_rl"
cd "$SCRIPTS"

REPO="$(cd "$BASE_DIR/.." && pwd)"
P1="${2:-}"
if [ -z "$P1" ]; then
  # 1) bundled phase1 checkpoint shipped with the repo (models/), used for all results
  BUNDLED="$REPO/models/phase1_mlp_s42/model_5999.pt"
  # 2) or a freshly-trained one from ./launch_phase1.sh (logs/)
  run=$(ls -dt logs/rsl_rl/unitree_go1_rough_teacher/*phase1_mlp_s${S} 2>/dev/null | head -1)
  if [ -f "$BUNDLED" ]; then P1="$BUNDLED"
  elif [ -n "$run" ]; then P1=$(ls "$run"/model_*.pt 2>/dev/null | awk -F'[_/.]' '{print $(NF-1)"\t"$0}' | sort -n | tail -1 | cut -f2); fi
fi
if [ -z "$P1" ] || [ ! -f "$P1" ]; then
  echo "ERROR: Phase 1 checkpoint not found. Bundled model missing and none trained."
  echo "       Run ./launch_phase1.sh $S first, or pass the path explicitly:"
  echo "       ./launch_warmstart_phase2.sh $S /path/to/model_5999.pt"
  exit 1
fi
echo "[phase2-warmstart] seed=$S  warmstart from: $P1"

GO1_NO_WARMSTART=0 PHASE1_CKPT="$P1" \
GO1_INJURY_ONEHOT=1 GO1_PROPRIO_ONLY=1 GO1_FLAT_TERRAIN=1 GO1_PHASE2_GAIT_TUNING=1 GO1_FEET_AIR_TIME_WEIGHT=0.1 \
GO1_PD_ACTUATOR=1 GO1_PD_KP=20.0 GO1_PD_KD=0.5 \
GO1_STRICT_TERMINATIONS=1 GO1_BAD_ORIENTATION_LIMIT=0.8 GO1_DOMAIN_RAND=1 \
GO1_CMD_VX_MIN=0.10 GO1_CMD_VX_MAX=0.30 GO1_CMD_VY_ABS=0.0 GO1_CMD_YAW_ABS=0.15 \
GO1_FLAT_ORIENTATION_WEIGHT=-2.0 GO1_SURVIVAL_BONUS_WEIGHT=1.0 \
GO1_SPLINT_CALF_ANGLE=-1.5 GO1_PEG_HIP_TORQUE_SCALE=1.0 GO1_SPLINT_CALF_STIFFNESS=12 GO1_SPLINT_CALF_DAMPING=1.0 \
GO1_INJURED_FORCE_NONUSE_WEIGHT=-1.0 GO1_INJURED_MIN_FORCE_SEVERE_N=5.0 GO1_INJURED_MIN_FORCE_MILD_N=5.0 \
GO1_INJURED_DUTY_NONUSE_WEIGHT=-1.5 GO1_INJURED_MIN_DUTY_SEVERE=0.35 GO1_INJURED_MIN_DUTY_MILD=0.35 \
GO1_INJURED_NONUSE_EMA_ALPHA=0.97 GO1_INJURED_NONUSE_RAMP_STEPS=40000 \
GO1_PAIN_WEIGHT=-0.05 GO1_BASE_HEIGHT_FLOOR_WEIGHT=0.0 GO1_PEG_GRACE_STEPS=30 \
GO1_CONTACT_FORCE_ASYM_WEIGHT=0.0 GO1_DUTY_FACTOR_ASYM_WEIGHT=0.0 GO1_DIAGONAL_LOAD_ASYM_WEIGHT=0.0 \
GO1_FRONT_REAR_LOAD_DIST_WEIGHT=0.0 GO1_TROT_SYNC_WEIGHT=0.0 \
PHASE2_RUN_NAME="phase2_warmstart_s${S}" PHASE2_MAX_ITER="${PHASE2_MAX_ITER:-12000}" \
NUM_ENVS="${NUM_ENVS:-2048}" SEED="$S" \
bash ./train_phase2.sh
