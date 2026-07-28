#!/bin/bash
# Train ONE paradigm teacher (foreground). Self-contained / portable: paths are
# resolved relative to this repo, no systemd, no hardcoded conda. Activate your
# Isaac Lab python env BEFORE running (so `python3 train.py` is the Isaac python).
#
# Three paradigms share an IDENTICAL environment, architecture, PPO, injury model
# (functional splint), PD control, domain randomisation, curriculum and the
# load-bearing viability floor. The ONLY difference is the impaired-limb reward:
#   antalgic : nociceptor pain penalty         (GO1_PAIN_WEIGHT=-0.05, SYM=0)
#   faulttol : no pain, alive bonus only        (GO1_PAIN_WEIGHT=0,    SYM=0)
#   symmetry : no pain, mirror-symmetry penalty (GO1_PAIN_WEIGHT=0,    SYM<0)
#
# Usage: ./launch_compar.sh <antalgic|faulttol|symmetry> <SEED>
set -uo pipefail
PARADIGM="${1:?paradigm}"; S="${2:?seed}"
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS="$(cd "$BASE_DIR/.." && pwd)/scripts/rsl_rl"

case "$PARADIGM" in
  antalgic)  PAIN=-0.05; SYM=0.0  ;;
  faulttol)  PAIN=0.0;   SYM=0.0  ;;
  symmetry)  PAIN=0.0;   SYM=-2.0 ;;
  *) echo "PARADIGM must be antalgic|faulttol|symmetry"; exit 1 ;;
esac
echo "[compar] $PARADIGM seed=$S  PAIN=$PAIN SYM=$SYM  scripts=$SCRIPTS"

cd "$SCRIPTS"
GO1_NO_WARMSTART=1 GO1_INJURY_ONEHOT=1 \
GO1_PD_ACTUATOR=1 GO1_PD_KP=20.0 GO1_PD_KD=0.5 \
GO1_STRICT_TERMINATIONS=1 GO1_BAD_ORIENTATION_LIMIT=0.8 GO1_DOMAIN_RAND=1 \
GO1_CMD_VX_MIN=0.10 GO1_CMD_VX_MAX=0.30 GO1_CMD_VY_ABS=0.0 GO1_CMD_YAW_ABS=0.15 \
GO1_FLAT_ORIENTATION_WEIGHT=-2.0 GO1_SURVIVAL_BONUS_WEIGHT=1.0 \
GO1_SPLINT_CALF_ANGLE=-1.5 GO1_PEG_HIP_TORQUE_SCALE=1.0 GO1_SPLINT_CALF_STIFFNESS=12 GO1_SPLINT_CALF_DAMPING=1.0 \
GO1_INJURED_FORCE_NONUSE_WEIGHT=-0.5 GO1_INJURED_MIN_FORCE_SEVERE_N=4.0 GO1_INJURED_MIN_FORCE_MILD_N=4.0 \
GO1_INJURED_DUTY_NONUSE_WEIGHT=-1.0 GO1_INJURED_MIN_DUTY_SEVERE=0.30 GO1_INJURED_MIN_DUTY_MILD=0.30 \
GO1_INJURED_NONUSE_EMA_ALPHA=0.97 GO1_INJURED_NONUSE_RAMP_STEPS=40000 \
GO1_BASE_HEIGHT_FLOOR_WEIGHT=0.0 GO1_PEG_GRACE_STEPS=30 \
GO1_PAIN_WEIGHT=$PAIN GO1_SYMMETRY_PENALTY_WEIGHT=$SYM \
PHASE2_RUN_NAME=phase2_cmp_${PARADIGM}_s$S PHASE2_MAX_ITER="${PHASE2_MAX_ITER:-18000}" \
NUM_ENVS="${NUM_ENVS:-2048}" SEED=$S \
bash ./train_phase2.sh
