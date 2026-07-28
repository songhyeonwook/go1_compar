#!/bin/bash
# Evaluate the 3 paradigms (antalgic/faulttol/symmetry) at low speed (straight)
# and print the comparison table. Self-contained / portable.
#
# Reads the runs produced by launch_warmstart_compar.sh (phase2_ws_<paradigm>_s<seed>).
#
# Usage: ./eval_compar.sh [SEED]   (default 42)
set -uo pipefail
S="${1:-42}"
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$BASE_DIR/.." && pwd)"
SCRIPTS="$REPO/scripts/rsl_rl"
OUT="$BASE_DIR/compare_result.txt"; : > "$OUT"
cd "$SCRIPTS"

latest () { ls "$1"/model_*.pt 2>/dev/null | awk -F'[_/.]' '{print $(NF-1)"\t"$0}' | sort -n | tail -1 | cut -f2-; }

ARGS=()
for P in antalgic faulttol symmetry; do
  run=$(ls -dt logs/rsl_rl/unitree_go1_rough_teacher/*phase2_ws_${P}_s${S} 2>/dev/null | head -1)
  ck=$([ -n "$run" ] && latest "$run")
  [ -z "$ck" ] && { echo "$P: no checkpoint (train it first: ./launch_warmstart_compar.sh $P $S)" >> "$OUT"; continue; }
  echo "$P: $(basename "$run")/$(basename "$ck")" >> "$OUT"
  rm -f "biomech/cmp_$P.npz"
  GO1_INJURY_ONEHOT=1 GO1_PROPRIO_ONLY=1 GO1_FLAT_TERRAIN=1 GO1_PEG_WEAKEN_JOINTS=hip \
  GO1_STRICT_TERMINATIONS=1 GO1_BAD_ORIENTATION_LIMIT=0.8 \
  GO1_SPLINT_CALF_ANGLE=-1.5 GO1_SPLINT_CALF_STIFFNESS=12 GO1_SPLINT_CALF_DAMPING=1.0 GO1_PEG_HIP_TORQUE_SCALE=1.0 \
  GO1_PD_ACTUATOR=1 GO1_PD_KP=20.0 GO1_PD_KD=0.5 GO1_BIOMECH_DUMP="$SCRIPTS/biomech/cmp_$P.npz" \
  GO1_CMD_VY_ABS=0.0 GO1_CMD_YAW_ABS=0.0 \
  TARGET_VX=0.3 CHECKPOINT="$ck" AGENT=rsl_rl_teacher_mlp_cfg_entry_point NUM_ENVS=400 STEPS=500 SEED=7 \
  bash ./analyze_phase2_balanced.sh > "$BASE_DIR/eval_$P.log" 2>&1 || true
  [ -f "biomech/cmp_$P.npz" ] && ARGS+=("$P=$SCRIPTS/biomech/cmp_$P.npz")
done

echo "" >> "$OUT"
python3 "$BASE_DIR/compare_3paradigm.py" "${ARGS[@]}" >> "$OUT" 2>&1
echo "ALLDONE" >> "$OUT"
cat "$OUT"
