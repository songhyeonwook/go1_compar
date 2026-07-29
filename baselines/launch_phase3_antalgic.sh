#!/bin/bash
# Phase 3 of the pipeline: STUDENT DISTILLATION for the PROPOSED antalgic method.
#
# A frozen Phase 2 antalgic teacher (privileged obs: injury one-hot + splint
# length + foot friction) is distilled into a recurrent (LSTM) student that sees
# ONLY proprioception (R^48). The student's adaptation module must INFER the
# hidden injury state -- which leg, splint length, foot friction -- from the
# proprioceptive history, and reproduce the teacher's antalgic action.
#
# Design invariant: the student MUST run in the SAME environment the teacher was
# trained/deployed in, or the frozen teacher (used as the distillation target)
# behaves differently than it learned. This launcher therefore mirrors the
# antalgic env block of launch_warmstart_compar.sh EXACTLY (PD actuator + gains,
# splint geometry, command range, terrain, domain rand, injury one-hot) and
# routes through the SAME entry point (train_phase2.sh, GO1_PHASE=student), so
# the env config cannot drift between teacher and student.
#
# Two deliberate differences from Phase 2 (both correct for distillation, not RL):
#   * curriculum OFF (GO1_USE_PEG_LEG_CURRICULUM=0): distillation is imitation,
#     not exploration, so the student should see the FULL deployment distribution
#     from step 0 -- injury prob 0.5 and the full splint range [0.20, 0.30] m --
#     rather than the teacher's easy->hard training ramp.
#   * reward knobs (pain, viability floor, symmetry) are still set for env
#     fidelity but are UNUSED: the student learns by MSE on the teacher latent,
#     not from reward.
#
# Self-contained / portable. Activate your Isaac Lab python env BEFORE running.
# Usage: ./launch_phase3_antalgic.sh <SEED> [PHASE2_TEACHER_CKPT]
# Output: <repo>/scripts/rsl_rl/logs/rsl_rl/unitree_go1_rough_student/*phase3_ws_antalgic_s<SEED>/
set -uo pipefail
S="${1:?seed}"
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$BASE_DIR/.." && pwd)"; SCRIPTS="$REPO/scripts/rsl_rl"
cd "$SCRIPTS"

# --- resolve the frozen Phase 2 antalgic teacher checkpoint -------------------
# Explicit 2nd arg wins; otherwise pick the latest model_*.pt from the matching
# Phase 2 antalgic run (phase2_ws_antalgic_s<SEED>).
T2="${2:-}"
if [ -z "$T2" ]; then
  run=$(ls -dt logs/rsl_rl/unitree_go1_rough_teacher/*phase2_ws_antalgic_s${S} 2>/dev/null | head -1)
  if [ -n "$run" ]; then
    T2=$(ls "$run"/model_*.pt 2>/dev/null | awk -F'[_/.]' '{print $(NF-1)"\t"$0}' | sort -n | tail -1 | cut -f2)
  fi
fi
[ -f "$T2" ] || { echo "ERROR: Phase 2 antalgic teacher ckpt not found. Train it first with ./launch_warmstart_compar.sh antalgic $S, or pass the path as arg 2."; exit 1; }
echo "[phase3-antalgic] seed=$S  distilling from teacher: $T2"

GO1_PHASE=student TEACHER_CKPT="$T2" \
GO1_INJURY_ONEHOT=1 GO1_PROPRIO_ONLY=1 GO1_FLAT_TERRAIN=1 GO1_PHASE2_GAIT_TUNING=1 GO1_FEET_AIR_TIME_WEIGHT=0.1 \
GO1_PD_ACTUATOR=1 GO1_PD_KP=20.0 GO1_PD_KD=0.5 \
GO1_STRICT_TERMINATIONS=1 GO1_BAD_ORIENTATION_LIMIT=0.8 GO1_DOMAIN_RAND=1 \
GO1_CMD_VX_MIN=0.10 GO1_CMD_VX_MAX=1.0 GO1_CMD_VY_ABS=0.0 GO1_CMD_YAW_ABS=0.15 \
GO1_FLAT_ORIENTATION_WEIGHT=-2.0 GO1_SURVIVAL_BONUS_WEIGHT=1.0 \
GO1_PEG_HIP_TORQUE_SCALE=1.0 GO1_SPLINT_CALF_STIFFNESS=12 GO1_SPLINT_CALF_DAMPING=1.0 \
GO1_PROB_PEG_LEG=0.5 GO1_USE_PEG_LEG_CURRICULUM=0 \
GO1_INJURED_FORCE_NONUSE_WEIGHT=-1.0 GO1_INJURED_MIN_FORCE_SEVERE_N=5.0 GO1_INJURED_MIN_FORCE_MILD_N=5.0 \
GO1_INJURED_DUTY_NONUSE_WEIGHT=-1.5 GO1_INJURED_MIN_DUTY_SEVERE=0.35 GO1_INJURED_MIN_DUTY_MILD=0.35 \
GO1_INJURED_NONUSE_EMA_ALPHA=0.97 GO1_INJURED_NONUSE_RAMP_STEPS=40000 \
GO1_PAIN_WEIGHT=-0.02 GO1_SYMMETRY_PENALTY_WEIGHT=0.0 GO1_BASE_HEIGHT_FLOOR_WEIGHT=0.0 GO1_PEG_GRACE_STEPS=30 \
GO1_CONTACT_FORCE_ASYM_WEIGHT=0.0 GO1_DUTY_FACTOR_ASYM_WEIGHT=0.0 GO1_DIAGONAL_LOAD_ASYM_WEIGHT=0.0 GO1_FRONT_REAR_LOAD_DIST_WEIGHT=0.0 GO1_TROT_SYNC_WEIGHT=0.0 \
PHASE3_RUN_NAME="phase3_ws_antalgic_s${S}" PHASE3_MAX_ITER="${PHASE3_MAX_ITER:-12000}" \
NUM_ENVS="${NUM_ENVS:-2048}" SEED="$S" \
bash ./train_phase2.sh
