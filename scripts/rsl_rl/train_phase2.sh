#!/bin/bash
# =============================================================================
# Phase 2 teacher training — THE single entry point (train.py wrapper).
# =============================================================================
# Every knob below is `${VAR:-default}`, so a launcher in baselines/ overrides it
# simply by exporting it. The defaults here are the paper configuration:
#
#   reward (eq.3)  r = W_task r_task(v,v*) - W_energy ||tau||^2 - W_pain C_pain(Fz)
#   pain   (eq.4)  C_pain(Fz) = Pbase 1[contact] + max(0, exp(a (Fz - Fth)) - 1)
#                  Pbase = 0.05, Fth = 10 N, a = 2.0   (FIXED, never severity-scaled)
#
# The prescriptive gait terms (contact/duty/diagonal asymmetry, trot_sync,
# front-rear load, leg duty target, intact overload) are all 0 by default: the
# paper's claim is that the antalgic pattern EMERGES from eq.3 alone, so nothing
# may prescribe a target gait, duty factor or asymmetry. Severity is handled by
# MORPHOLOGY (the splint's eq.1-2 Jacobian), not by scaling the pain function.
#
# The one exception is the load-bearing viability floor
# (GO1_INJURED_*_NONUSE_*): it prevents total non-use of the impaired limb so
# the paradigms are compared on gait PATTERN rather than use-vs-nonuse. It is
# applied identically to every paradigm — document this choice in the paper.
#
# Balance shaping (flat_orientation + relaxed base height) exists because a
# short rigid peg otherwise reaches the ground by ROLLING the trunk into the
# injured corner until it tips over; penalising trunk-tilt ANGLE and allowing a
# slight squat lets it load the peg at a LEVEL posture instead.
#
# Usage (normally called via baselines/launch_*.sh, not directly):
#   GO1_NO_WARMSTART=1 PHASE2_RUN_NAME=my_run ./train_phase2.sh   # from scratch
#   PHASE1_CKPT=/path/model_5999.pt PHASE2_RUN_NAME=my_run ./train_phase2.sh
# =============================================================================

set -euo pipefail
TRAIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$TRAIN_DIR"

# --- run identity ------------------------------------------------------------
TASK="${TASK:-Template-Go1-Lab-v0}"
EXP_NAME="${EXP_NAME:-unitree_go1_rough_teacher}"
RUN_NAME="${PHASE2_RUN_NAME:-phase2_teacher}"
export AGENT="${AGENT:-rsl_rl_teacher_mlp_cfg_entry_point}"
NUM_ENVS="${NUM_ENVS:-4096}"
MAX_ITER="${PHASE2_MAX_ITER:-6000}"
SEED="${SEED:-42}"

# --- warm-start --------------------------------------------------------------
# GO1_NO_WARMSTART=1 trains FROM SCRATCH, so an asymmetric injured gait can
# emerge instead of being trapped in the symmetric-trot basin of a healthy
# warm-start. Otherwise PHASE1_CKPT must point at the Phase 1 checkpoint
# (baselines/launch_warmstart_*.sh resolve it, defaulting to models/).
PHASE1_CKPT="${PHASE1_CKPT:-}"
if [ "${GO1_NO_WARMSTART:-0}" = "1" ]; then
    WS_ARGS=()
    PHASE1_CKPT="(none: from scratch)"
elif [ -z "$PHASE1_CKPT" ]; then
    echo "ERROR: warm-start requested but PHASE1_CKPT is empty."
    echo "       set PHASE1_CKPT=/path/to/model_N.pt, or GO1_NO_WARMSTART=1 for from-scratch."
    exit 1
elif [ ! -f "$PHASE1_CKPT" ]; then
    echo "ERROR: Phase 1 checkpoint not found: $PHASE1_CKPT"
    exit 1
else
    WS_ARGS=(--warmstart_ckpt_path "$PHASE1_CKPT")
fi

# --- environment / observation ----------------------------------------------
# Proprioception-only R^48 on flat terrain (paper §4.3: "no exteroceptive
# sensing"); privileged obs stay on for the teacher. Phase 1 must use the same
# flags or the warm-start is dimension-incompatible.
export GO1_PHASE="${GO1_PHASE:-teacher}"
export GO1_EVAL_MODE="${GO1_EVAL_MODE:-random}"
export GO1_PROPRIO_ONLY="${GO1_PROPRIO_ONLY:-1}"
export GO1_FLAT_TERRAIN="${GO1_FLAT_TERRAIN:-1}"
export GO1_DOMAIN_RAND="${GO1_DOMAIN_RAND:-1}"
export GO1_PHASE1_BALANCE_REWARDS="${GO1_PHASE1_BALANCE_REWARDS:-0}"
export GO1_PHASE2_GAIT_TUNING="${GO1_PHASE2_GAIT_TUNING:-0}"

# --- injury model (functional splint) + curriculum ---------------------------
export GO1_PEG_WEAKEN_JOINTS="${GO1_PEG_WEAKEN_JOINTS:-hip}"
export GO1_PROB_PEG_LEG="${GO1_PROB_PEG_LEG:-0.5}"
export GO1_SPLINT_LENGTH_MIN="${GO1_SPLINT_LENGTH_MIN:-0.20}"
export GO1_SPLINT_LENGTH_MAX="${GO1_SPLINT_LENGTH_MAX:-0.30}"
export GO1_FOOT_FRICTION_MIN="${GO1_FOOT_FRICTION_MIN:-0.5}"
export GO1_FOOT_FRICTION_MAX="${GO1_FOOT_FRICTION_MAX:-1.5}"
export GO1_USE_PEG_LEG_CURRICULUM="${GO1_USE_PEG_LEG_CURRICULUM:-1}"
export GO1_CURRICULUM_PROB_START="${GO1_CURRICULUM_PROB_START:-0.1}"
export GO1_CURRICULUM_PROB_RAMP_STEPS="${GO1_CURRICULUM_PROB_RAMP_STEPS:-5000}"
export GO1_CURRICULUM_SPLINT_LO_START="${GO1_CURRICULUM_SPLINT_LO_START:-0.28}"
export GO1_CURRICULUM_SPLINT_HI_START="${GO1_CURRICULUM_SPLINT_HI_START:-0.33}"
export GO1_CURRICULUM_SPLINT_RAMP_STEPS="${GO1_CURRICULUM_SPLINT_RAMP_STEPS:-8000}"

# --- pain, eq.4 (the ONLY injury-specific reward; W_pain is the sole knob) ----
export GO1_PAIN_WEIGHT="${GO1_PAIN_WEIGHT:--0.10}"
export GO1_PAIN_THRESHOLD_N="${GO1_PAIN_THRESHOLD_N:-10.0}"
export GO1_PAIN_SCALE="${GO1_PAIN_SCALE:-2.0}"
export GO1_PAIN_BASE_CONTACT_COST="${GO1_PAIN_BASE_CONTACT_COST:-0.05}"
export GO1_PAIN_OVERLOAD_TOLERANCE="${GO1_PAIN_OVERLOAD_TOLERANCE:-0.0}"
export GO1_PAIN_MAX_EXP_ARGUMENT="${GO1_PAIN_MAX_EXP_ARGUMENT:-10}"
export GO1_PAIN_MAX_PENALTY="${GO1_PAIN_MAX_PENALTY:-200}"
# the rigid stump contacts ground through the calf/peg, so Fz at the impaired
# limb IS the calf contact force -> include_calf=1 is the correct measurement.
export GO1_PAIN_INCLUDE_CALF="${GO1_PAIN_INCLUDE_CALF:-1}"
export GO1_PAIN_LOAD_CONTACT_COST="${GO1_PAIN_LOAD_CONTACT_COST:-0.0}"
export GO1_PAIN_SEVERITY_SCALING="${GO1_PAIN_SEVERITY_SCALING:-0}"
export GO1_PAIN_LOAD_COST_SEVERE_MULT="${GO1_PAIN_LOAD_COST_SEVERE_MULT:-1.20}"
export GO1_PAIN_LOAD_COST_MILD_MULT="${GO1_PAIN_LOAD_COST_MILD_MULT:-0.80}"
export GO1_PAIN_THRESHOLD_SEVERE_MULT="${GO1_PAIN_THRESHOLD_SEVERE_MULT:-0.80}"
export GO1_PAIN_THRESHOLD_MILD_MULT="${GO1_PAIN_THRESHOLD_MILD_MULT:-1.15}"
export GO1_PAIN_SCALE_SEVERE_MULT="${GO1_PAIN_SCALE_SEVERE_MULT:-1.25}"
export GO1_PAIN_SCALE_MILD_MULT="${GO1_PAIN_SCALE_MILD_MULT:-0.85}"
export GO1_LOAD_CONTACT_THRESHOLD_N="${GO1_LOAD_CONTACT_THRESHOLD_N:-10.0}"

# --- alternative impaired-limb rewards (the 3-paradigm comparison variable) ---
# antalgic = pain only; faulttol = neither; symmetry = mirror penalty only.
export GO1_SYMMETRY_PENALTY_WEIGHT="${GO1_SYMMETRY_PENALTY_WEIGHT:-0.0}"

# --- load-bearing viability floor (identical across paradigms) ----------------
export GO1_INJURED_FORCE_NONUSE_WEIGHT="${GO1_INJURED_FORCE_NONUSE_WEIGHT:--1.0}"
export GO1_INJURED_MIN_FORCE_SEVERE_N="${GO1_INJURED_MIN_FORCE_SEVERE_N:-6.0}"
export GO1_INJURED_MIN_FORCE_MILD_N="${GO1_INJURED_MIN_FORCE_MILD_N:-6.0}"
export GO1_INJURED_MIN_FORCE_FRONT_MULT="${GO1_INJURED_MIN_FORCE_FRONT_MULT:-1.15}"
export GO1_INJURED_MIN_FORCE_REAR_MULT="${GO1_INJURED_MIN_FORCE_REAR_MULT:-1.0}"
export GO1_INJURED_DUTY_NONUSE_WEIGHT="${GO1_INJURED_DUTY_NONUSE_WEIGHT:-0.0}"
export GO1_INJURED_MIN_DUTY_SEVERE="${GO1_INJURED_MIN_DUTY_SEVERE:-0.05}"
export GO1_INJURED_MIN_DUTY_MILD="${GO1_INJURED_MIN_DUTY_MILD:-0.28}"
export GO1_INJURED_MIN_DUTY_FRONT_MULT="${GO1_INJURED_MIN_DUTY_FRONT_MULT:-1.10}"
export GO1_INJURED_MIN_DUTY_REAR_MULT="${GO1_INJURED_MIN_DUTY_REAR_MULT:-1.0}"
export GO1_INJURED_NONUSE_EMA_ALPHA="${GO1_INJURED_NONUSE_EMA_ALPHA:-0.90}"
export GO1_INJURED_NONUSE_RAMP_STEPS="${GO1_INJURED_NONUSE_RAMP_STEPS:-2000}"

# --- task / energy / balance shaping -----------------------------------------
export GO1_TRACK_LIN_VEL_WEIGHT="${GO1_TRACK_LIN_VEL_WEIGHT:-2.5}"
export GO1_TORQUE_PENALTY_SCALE="${GO1_TORQUE_PENALTY_SCALE:-30}"
export GO1_SURVIVAL_BONUS_WEIGHT="${GO1_SURVIVAL_BONUS_WEIGHT:-0.5}"
export GO1_FLAT_ORIENTATION_WEIGHT="${GO1_FLAT_ORIENTATION_WEIGHT:--1.5}"
export GO1_BASE_HEIGHT_TARGET="${GO1_BASE_HEIGHT_TARGET:-0.27}"
export GO1_BASE_HEIGHT_WEIGHT="${GO1_BASE_HEIGHT_WEIGHT:--0.15}"
export GO1_ROOT_TOO_LOW_MIN_HEIGHT="${GO1_ROOT_TOO_LOW_MIN_HEIGHT:-0.15}"
export GO1_FRONT_PAYLOAD_KG="${GO1_FRONT_PAYLOAD_KG:-0.0}"
export GO1_FRONT_COM_X_M="${GO1_FRONT_COM_X_M:-0.0}"

# --- prescriptive gait terms: OFF (would contradict the emergence claim) ------
export GO1_CONTACT_FORCE_ASYM_WEIGHT="${GO1_CONTACT_FORCE_ASYM_WEIGHT:-0.0}"
export GO1_DUTY_FACTOR_ASYM_WEIGHT="${GO1_DUTY_FACTOR_ASYM_WEIGHT:-0.0}"
export GO1_DIAGONAL_LOAD_ASYM_WEIGHT="${GO1_DIAGONAL_LOAD_ASYM_WEIGHT:-0.0}"
export GO1_FRONT_REAR_LOAD_DIST_WEIGHT="${GO1_FRONT_REAR_LOAD_DIST_WEIGHT:-0.0}"
export GO1_TROT_SYNC_WEIGHT="${GO1_TROT_SYNC_WEIGHT:-0.0}"
export GO1_LEG_DUTY_TARGET_WEIGHT="${GO1_LEG_DUTY_TARGET_WEIGHT:-0.0}"
export GO1_DUTY_FACTOR_DEVIATION_WEIGHT="${GO1_DUTY_FACTOR_DEVIATION_WEIGHT:-0.0}"
export GO1_INTACT_OVERLOAD_WEIGHT="${GO1_INTACT_OVERLOAD_WEIGHT:-0.0}"
export GO1_INJURED_DUTY_OVERUSE_WEIGHT="${GO1_INJURED_DUTY_OVERUSE_WEIGHT:-0.0}"
export GO1_INJURED_LIGHT_DRAG_WEIGHT="${GO1_INJURED_LIGHT_DRAG_WEIGHT:-0.0}"
# ramp/threshold params for the terms above; inert while their weights are 0.
export GO1_CONTACT_FORCE_ASYM_RAMP_STEPS="${GO1_CONTACT_FORCE_ASYM_RAMP_STEPS:-6000}"
export GO1_DUTY_FACTOR_ASYM_RAMP_STEPS="${GO1_DUTY_FACTOR_ASYM_RAMP_STEPS:-6000}"
export GO1_DIAGONAL_LOAD_ASYM_RAMP_STEPS="${GO1_DIAGONAL_LOAD_ASYM_RAMP_STEPS:-6000}"
export GO1_FRONT_REAR_LOAD_DIST_RAMP_STEPS="${GO1_FRONT_REAR_LOAD_DIST_RAMP_STEPS:-6000}"
export GO1_TROT_SYNC_RAMP_STEPS="${GO1_TROT_SYNC_RAMP_STEPS:-6000}"
export GO1_FRONT_LOAD_TARGET_FRACTION="${GO1_FRONT_LOAD_TARGET_FRACTION:-0.55}"
export GO1_FRONT_LOAD_TARGET_TOLERANCE="${GO1_FRONT_LOAD_TARGET_TOLERANCE:-0.06}"
export GO1_INJURED_OVERUSE_RAMP_STEPS="${GO1_INJURED_OVERUSE_RAMP_STEPS:-8000}"
export GO1_INJURED_MAX_DUTY_SEVERE="${GO1_INJURED_MAX_DUTY_SEVERE:-0.30}"
export GO1_INJURED_MAX_DUTY_MILD="${GO1_INJURED_MAX_DUTY_MILD:-0.50}"
export GO1_INJURED_MAX_DUTY_FRONT_MULT="${GO1_INJURED_MAX_DUTY_FRONT_MULT:-0.95}"
export GO1_INJURED_MAX_DUTY_REAR_MULT="${GO1_INJURED_MAX_DUTY_REAR_MULT:-1.0}"
export GO1_INTACT_OVERLOAD_THRESHOLD_N="${GO1_INTACT_OVERLOAD_THRESHOLD_N:-65.0}"
export GO1_INTACT_OVERLOAD_SCALE="${GO1_INTACT_OVERLOAD_SCALE:-1.0}"
export GO1_INTACT_OVERLOAD_MAX_PENALTY="${GO1_INTACT_OVERLOAD_MAX_PENALTY:-120.0}"

echo "-----------------------------------------------"
echo "  Phase 2: peg-leg teacher training"
echo "  run_name=$RUN_NAME  agent=$AGENT"
echo "  warmstart=$PHASE1_CKPT"
echo "  num_envs=$NUM_ENVS max_iterations=$MAX_ITER seed=$SEED"
echo "  peg_leg_prob=$GO1_PROB_PEG_LEG curriculum=$GO1_USE_PEG_LEG_CURRICULUM"
echo "  splint_range=[$GO1_SPLINT_LENGTH_MIN, $GO1_SPLINT_LENGTH_MAX] m"
echo "  pain: weight=$GO1_PAIN_WEIGHT threshold=${GO1_PAIN_THRESHOLD_N}N scale=$GO1_PAIN_SCALE base=$GO1_PAIN_BASE_CONTACT_COST"
echo "  symmetry_penalty=$GO1_SYMMETRY_PENALTY_WEIGHT"
echo "  viability floor: force=$GO1_INJURED_FORCE_NONUSE_WEIGHT@${GO1_INJURED_MIN_FORCE_SEVERE_N}N duty=$GO1_INJURED_DUTY_NONUSE_WEIGHT@$GO1_INJURED_MIN_DUTY_SEVERE ramp=$GO1_INJURED_NONUSE_RAMP_STEPS ema=$GO1_INJURED_NONUSE_EMA_ALPHA"
echo "  balance: flat_orientation=$GO1_FLAT_ORIENTATION_WEIGHT base_height=$GO1_BASE_HEIGHT_TARGET@$GO1_BASE_HEIGHT_WEIGHT"
echo "  gait_tuning=$GO1_PHASE2_GAIT_TUNING"
echo "-----------------------------------------------"

python3 train.py \
    --task "$TASK" \
    --agent "$AGENT" \
    --num_envs "$NUM_ENVS" \
    --headless \
    --experiment_name "$EXP_NAME" \
    --run_name "$RUN_NAME" \
    "${WS_ARGS[@]}" \
    --max_iterations "$MAX_ITER" \
    --seed "$SEED" \
    --use_peg_leg_action_mask

echo ""
echo "-----------------------------------------------"
echo "  Phase 2 complete"
echo "  Logs: $TRAIN_DIR/logs/rsl_rl/$EXP_NAME"
echo "-----------------------------------------------"
