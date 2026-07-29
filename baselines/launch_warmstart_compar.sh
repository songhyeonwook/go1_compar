#!/bin/bash
# Phase 2 of the pipeline, WARM-STARTED from the SAME clean phase1
# (models/phase1_mlp_s42). This trains BOTH the proposed method and its controls --
# they are the same pipeline and differ only in the impaired-limb reward:
#   antalgic : nociceptor pain (exp, eq.4)      GO1_PAIN_WEIGHT=-0.02  SYM=0   <-- PROPOSED
#   faulttol : no pain, alive bonus only        GO1_PAIN_WEIGHT=0      SYM=0       control
#   symmetry : no pain, mirror-symmetry penalty GO1_PAIN_WEIGHT=0      SYM=-2.0    control
# Maximally controlled: identical phase1 init, environment, injury model, viability
# floor, gait_tuning, PD and schedule.
# Warm-starting all three from the same symmetric healthy gait is a CONSERVATIVE test
# for antalgia: the policy must actively break the symmetry it inherited in order to
# off-load the impaired limb, and every paradigm starts from the same physiological
# baseline rather than from three unrelated random inits.
#
# Self-contained / portable. Activate your Isaac Lab python env BEFORE running.
# Usage: ./launch_warmstart_compar.sh <antalgic|faulttol|symmetry> <SEED> [PHASE1_CKPT]
set -uo pipefail
PARADIGM="${1:?paradigm}"; S="${2:?seed}"
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$BASE_DIR/.." && pwd)"; SCRIPTS="$REPO/scripts/rsl_rl"
cd "$SCRIPTS"

case "$PARADIGM" in
  antalgic) PAIN=-0.02; SYM=0.0 ;;   # -0.02 = mildest pain that off-loads (84%) with load conserved (dose-response)
  faulttol) PAIN=0.0;   SYM=0.0 ;;
  symmetry) PAIN=0.0;   SYM=-2.0 ;;
  *) echo "PARADIGM must be antalgic|faulttol|symmetry"; exit 1 ;;
esac

P1="${3:-}"
if [ -z "$P1" ]; then
  BUNDLED="$REPO/models/phase1_mlp_s42/model_5999.pt"
  run=$(ls -dt logs/rsl_rl/unitree_go1_rough_teacher/*phase1_mlp_s${S} 2>/dev/null | head -1)
  if [ -f "$BUNDLED" ]; then P1="$BUNDLED"
  elif [ -n "$run" ]; then P1=$(ls "$run"/model_*.pt 2>/dev/null | awk -F'[_/.]' '{print $(NF-1)"\t"$0}' | sort -n | tail -1 | cut -f2); fi
fi
[ -f "$P1" ] || { echo "ERROR: phase1 ckpt not found (bundled missing, none trained). Run ./launch_phase1.sh first."; exit 1; }
echo "[warmstart-compar] $PARADIGM seed=$S  PAIN=$PAIN SYM=$SYM  warmstart from: $P1"

GO1_NO_WARMSTART=0 PHASE1_CKPT="$P1" \
GO1_INJURY_ONEHOT=1 GO1_PROPRIO_ONLY=1 GO1_FLAT_TERRAIN=1 GO1_PHASE2_GAIT_TUNING=1 GO1_FEET_AIR_TIME_WEIGHT=0.1 \
GO1_PD_ACTUATOR=1 GO1_PD_KP=20.0 GO1_PD_KD=0.5 \
GO1_STRICT_TERMINATIONS=1 GO1_BAD_ORIENTATION_LIMIT=0.8 GO1_DOMAIN_RAND=1 \
GO1_CMD_VX_MIN=0.10 GO1_CMD_VX_MAX=1.0 GO1_CMD_VY_ABS=0.0 GO1_CMD_YAW_ABS=0.15 \
GO1_FLAT_ORIENTATION_WEIGHT=-2.0 GO1_SURVIVAL_BONUS_WEIGHT=1.0 \
GO1_PEG_HIP_TORQUE_SCALE=1.0 GO1_SPLINT_CALF_STIFFNESS=12 GO1_SPLINT_CALF_DAMPING=1.0 \
GO1_INJURED_FORCE_NONUSE_WEIGHT=-1.0 GO1_INJURED_MIN_FORCE_SEVERE_N=5.0 GO1_INJURED_MIN_FORCE_MILD_N=5.0 \
GO1_INJURED_DUTY_NONUSE_WEIGHT=-1.5 GO1_INJURED_MIN_DUTY_SEVERE=0.35 GO1_INJURED_MIN_DUTY_MILD=0.35 \
GO1_INJURED_NONUSE_EMA_ALPHA=0.97 GO1_INJURED_NONUSE_RAMP_STEPS=40000 \
GO1_PAIN_WEIGHT=$PAIN GO1_SYMMETRY_PENALTY_WEIGHT=$SYM GO1_BASE_HEIGHT_FLOOR_WEIGHT=0.0 GO1_PEG_GRACE_STEPS=30 \
GO1_CONTACT_FORCE_ASYM_WEIGHT=0.0 GO1_DUTY_FACTOR_ASYM_WEIGHT=0.0 GO1_DIAGONAL_LOAD_ASYM_WEIGHT=0.0 GO1_FRONT_REAR_LOAD_DIST_WEIGHT=0.0 GO1_TROT_SYNC_WEIGHT=0.0 \
PHASE2_RUN_NAME="phase2_ws_${PARADIGM}_s${S}" PHASE2_MAX_ITER="${PHASE2_MAX_ITER:-12000}" \
NUM_ENVS="${NUM_ENVS:-2048}" SEED="$S" \
bash ./train_phase2.sh
