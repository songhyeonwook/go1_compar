#!/bin/bash
# Evaluate the 3 paradigms (antalgic/faulttol/symmetry) over a SPEED SWEEP and
# print one comparison table per commanded speed. Self-contained / portable.
#
# Reads the runs produced by launch_warmstart_compar.sh (phase2_ws_<paradigm>_s<seed>).
#
# GRF and duty factor are strongly speed-dependent, and the injured-animal
# literature reports them as a function of speed, so the paradigms are compared
# at each speed separately rather than pooled into one number.
#
# Usage: ./eval_compar.sh [SEED]        (default 42)
# Env:   SPEEDS="0.0 0.25 0.5 0.75 1.0" commanded vx values (m/s)
#        NUM_ENVS, STEPS
#
# NOTE: the policies are trained with GO1_CMD_VX_MIN/MAX (see
# launch_warmstart_compar.sh). Speeds outside that range are extrapolation --
# keep the sweep inside the trained range, or widen the training range to match.
set -uo pipefail
S="${1:-42}"
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$BASE_DIR/.." && pwd)"
SCRIPTS="$REPO/scripts/rsl_rl"
SPEEDS="${SPEEDS:-0.0 0.25 0.5 0.75 1.0}"
OUT="$BASE_DIR/compare_result.txt"; : > "$OUT"
cd "$SCRIPTS"

latest () { ls "$1"/model_*.pt 2>/dev/null | awk -F'[_/.]' '{print $(NF-1)"\t"$0}' | sort -n | tail -1 | cut -f2-; }
tag ()    { echo "$1" | tr '.' 'p'; }        # 0.25 -> 0p25 (filename-safe)

# ---- resolve one checkpoint per paradigm (shared across all speeds) ----
declare -A CK
for P in antalgic faulttol symmetry; do
  run=$(ls -dt logs/rsl_rl/unitree_go1_rough_teacher/*phase2_ws_${P}_s${S} 2>/dev/null | head -1)
  ck=$([ -n "$run" ] && latest "$run")
  if [ -z "$ck" ]; then
    echo "$P: no checkpoint (train it first: ./launch_warmstart_compar.sh $P $S)" >> "$OUT"
  else
    CK[$P]="$ck"
    echo "$P: $(basename "$run")/$(basename "$ck")" >> "$OUT"
  fi
done
echo "" >> "$OUT"
echo "speed sweep: $SPEEDS m/s" >> "$OUT"

# ---- evaluate every (paradigm, speed) and compare per speed ----
for V in $SPEEDS; do
  VT=$(tag "$V")
  ARGS=()
  for P in antalgic faulttol symmetry; do
    ck="${CK[$P]:-}"; [ -z "$ck" ] && continue
    npz="$SCRIPTS/biomech/cmp_${P}_v${VT}.npz"
    rm -f "$npz"
    GO1_INJURY_ONEHOT=1 GO1_PROPRIO_ONLY=1 GO1_FLAT_TERRAIN=1 GO1_PEG_WEAKEN_JOINTS=hip \
    GO1_STRICT_TERMINATIONS=1 GO1_BAD_ORIENTATION_LIMIT=0.8 \
    GO1_SPLINT_CALF_STIFFNESS=100 GO1_SPLINT_CALF_DAMPING=1.0 GO1_PEG_HIP_TORQUE_SCALE=1.0 \
    GO1_PD_ACTUATOR=1 GO1_PD_KP=20.0 GO1_PD_KD=0.5 GO1_BIOMECH_DUMP="$npz" \
    GO1_CMD_VY_ABS=0.0 GO1_CMD_YAW_ABS=0.0 \
    TARGET_VX="$V" CHECKPOINT="$ck" AGENT=rsl_rl_teacher_mlp_cfg_entry_point \
    NUM_ENVS="${NUM_ENVS:-400}" STEPS="${STEPS:-500}" SEED=7 \
    bash ./analyze_phase2_balanced.sh > "$BASE_DIR/eval_${P}_v${VT}.log" 2>&1 || true
    [ -f "$npz" ] && ARGS+=("$P=$npz")
  done
  {
    echo ""
    echo "=============================================================="
    echo "  vx = $V m/s"
    echo "=============================================================="
  } >> "$OUT"
  if [ "${#ARGS[@]}" -eq 0 ]; then
    echo "  (no biomech dump at this speed — see eval_*_v${VT}.log)" >> "$OUT"
  else
    python3 "$BASE_DIR/compare_3paradigm.py" "${ARGS[@]}" >> "$OUT" 2>&1
  fi
done

echo "ALLDONE" >> "$OUT"
cat "$OUT"
