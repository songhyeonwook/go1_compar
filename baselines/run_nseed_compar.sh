#!/bin/bash
# n-seed 3-paradigm comparison: train antalgic/faulttol/symmetry for each seed,
# evaluate, and aggregate per paradigm (mean +/- SD + 95% CI). Self-contained.
#
#   nohup ./run_nseed_compar.sh "42 43 44 45 46" > nseed.log 2>&1 &
#
# Env: PARALLEL (concurrent trainings, default 3), PHASE2_MAX_ITER, NUM_ENVS.
set -uo pipefail
SEEDS="${1:-42 43 44 45 46}"
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$BASE_DIR/.." && pwd)"
SCRIPTS="$REPO/scripts/rsl_rl"
PARALLEL="${PARALLEL:-3}"
PARADIGMS="antalgic faulttol symmetry"
cd "$SCRIPTS"
latest () { ls "$1"/model_*.pt 2>/dev/null | awk -F'[_/.]' '{print $(NF-1)"\t"$0}' | sort -n | tail -1 | cut -f2-; }

# ---- 1. train every (paradigm, seed) ----
running=0
for S in $SEEDS; do
  for P in $PARADIGMS; do
    bash "$BASE_DIR/launch_compar.sh" "$P" "$S" > "$BASE_DIR/train_${P}_$S.log" 2>&1 &
    running=$((running+1))
    [ "$running" -ge "$PARALLEL" ] && { wait -n 2>/dev/null || wait; running=$((running-1)); }
  done
done
wait
echo "[nseed_compar] all trainings done"

# ---- 2. evaluate every (paradigm, seed) ----
for S in $SEEDS; do
  for P in $PARADIGMS; do
    run=$(ls -dt logs/rsl_rl/unitree_go1_rough_teacher/*phase2_cmp_${P}_s${S} 2>/dev/null | head -1)
    ck=$([ -n "$run" ] && latest "$run"); [ -z "$ck" ] && continue
    rm -f "biomech/cmp_${P}_s${S}.npz"
    GO1_INJURY_ONEHOT=1 GO1_PROPRIO_ONLY=1 GO1_FLAT_TERRAIN=1 GO1_PEG_WEAKEN_JOINTS=hip \
    GO1_STRICT_TERMINATIONS=1 GO1_BAD_ORIENTATION_LIMIT=0.8 \
    GO1_SPLINT_CALF_ANGLE=-1.5 GO1_SPLINT_CALF_STIFFNESS=12 GO1_SPLINT_CALF_DAMPING=1.0 GO1_PEG_HIP_TORQUE_SCALE=1.0 \
    GO1_PD_ACTUATOR=1 GO1_PD_KP=20.0 GO1_PD_KD=0.5 GO1_BIOMECH_DUMP="$SCRIPTS/biomech/cmp_${P}_s${S}.npz" \
    GO1_CMD_VY_ABS=0.0 GO1_CMD_YAW_ABS=0.0 \
    TARGET_VX=0.3 CHECKPOINT="$ck" AGENT=rsl_rl_teacher_mlp_cfg_entry_point NUM_ENVS=400 STEPS=500 SEED=7 \
    bash ./analyze_phase2_balanced.sh > "$BASE_DIR/eval_${P}_s${S}.log" 2>&1 || true
  done
done

# ---- 3. aggregate per paradigm ----
OUT="$BASE_DIR/nseed_compare_result.txt"; : > "$OUT"
for P in $PARADIGMS; do
  args=()
  for S in $SEEDS; do [ -f "biomech/cmp_${P}_s${S}.npz" ] && args+=("seed$S=$SCRIPTS/biomech/cmp_${P}_s${S}.npz"); done
  echo "===== $P  (n=${#args[@]}) =====" >> "$OUT"
  python3 "$SCRIPTS/aggregate_nseed.py" "${args[@]}" >> "$OUT" 2>&1
  echo "" >> "$OUT"
done
echo "ALLDONE" >> "$OUT"
cat "$OUT"
