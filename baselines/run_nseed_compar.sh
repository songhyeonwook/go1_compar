#!/bin/bash
# n-seed 3-paradigm comparison: train antalgic/faulttol/symmetry for each seed,
# evaluate, and aggregate per paradigm (mean +/- SD + 95% CI). Self-contained.
#
#   nohup ./run_nseed_compar.sh "42 43 44 45 46" > nseed.log 2>&1 &
#
# Every paradigm and every seed warm-starts from the same bundled phase1, so the
# spread reported here is phase2 training variance, NOT phase1 variance.
#
# Evaluated over a speed sweep; the aggregate is reported per (paradigm, speed),
# since GRF and duty factor are speed-dependent. Keep SPEEDS inside the trained
# GO1_CMD_VX_MIN/MAX range (see launch_warmstart_compar.sh) or it is extrapolation.
#
# Env: PARALLEL (concurrent trainings, default 3), PHASE2_MAX_ITER, NUM_ENVS,
#      SPEEDS="0.0 0.25 0.5 0.75 1.0".
set -uo pipefail
SEEDS="${1:-42 43 44 45 46}"
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$BASE_DIR/.." && pwd)"
SCRIPTS="$REPO/scripts/rsl_rl"
PARALLEL="${PARALLEL:-3}"
PARADIGMS="antalgic faulttol symmetry"
SPEEDS="${SPEEDS:-0.0 0.25 0.5 0.75 1.0}"
cd "$SCRIPTS"
latest () { ls "$1"/model_*.pt 2>/dev/null | awk -F'[_/.]' '{print $(NF-1)"\t"$0}' | sort -n | tail -1 | cut -f2-; }
tag ()    { echo "$1" | tr '.' 'p'; }

# ---- 1. train every (paradigm, seed) ----
running=0
for S in $SEEDS; do
  for P in $PARADIGMS; do
    bash "$BASE_DIR/launch_warmstart_compar.sh" "$P" "$S" > "$BASE_DIR/train_${P}_$S.log" 2>&1 &
    running=$((running+1))
    [ "$running" -ge "$PARALLEL" ] && { wait -n 2>/dev/null || wait; running=$((running-1)); }
  done
done
wait
echo "[nseed_compar] all trainings done"

# ---- 2. evaluate every (paradigm, seed, speed) ----
for S in $SEEDS; do
  for P in $PARADIGMS; do
    run=$(ls -dt logs/rsl_rl/unitree_go1_rough_teacher/*phase2_ws_${P}_s${S} 2>/dev/null | head -1)
    ck=$([ -n "$run" ] && latest "$run"); [ -z "$ck" ] && continue
    for V in $SPEEDS; do
      VT=$(tag "$V")
      npz="$SCRIPTS/biomech/cmp_${P}_s${S}_v${VT}.npz"
      rm -f "$npz"
      GO1_INJURY_ONEHOT=1 GO1_PROPRIO_ONLY=1 GO1_FLAT_TERRAIN=1 GO1_PEG_WEAKEN_JOINTS=hip \
      GO1_STRICT_TERMINATIONS=1 GO1_BAD_ORIENTATION_LIMIT=0.8 \
      GO1_SPLINT_CALF_STIFFNESS=100 GO1_SPLINT_CALF_DAMPING=1.0 GO1_PEG_HIP_TORQUE_SCALE=1.0 \
      GO1_PD_ACTUATOR=1 GO1_PD_KP=20.0 GO1_PD_KD=0.5 GO1_BIOMECH_DUMP="$npz" \
      GO1_CMD_VY_ABS=0.0 GO1_CMD_YAW_ABS=0.0 \
      TARGET_VX="$V" CHECKPOINT="$ck" AGENT=rsl_rl_teacher_mlp_cfg_entry_point \
      NUM_ENVS="${NUM_ENVS:-400}" STEPS="${STEPS:-500}" SEED=7 \
      bash ./analyze_phase2_balanced.sh > "$BASE_DIR/eval_${P}_s${S}_v${VT}.log" 2>&1 || true
    done
  done
done

# ---- 3. aggregate across seeds, per (paradigm, speed) ----
OUT="$BASE_DIR/nseed_compare_result.txt"; : > "$OUT"
echo "seeds: $SEEDS" >> "$OUT"
echo "speed sweep: $SPEEDS m/s" >> "$OUT"
for V in $SPEEDS; do
  VT=$(tag "$V")
  {
    echo ""
    echo "=============================================================="
    echo "  vx = $V m/s"
    echo "=============================================================="
  } >> "$OUT"
  for P in $PARADIGMS; do
    args=()
    for S in $SEEDS; do
      npz="biomech/cmp_${P}_s${S}_v${VT}.npz"
      [ -f "$npz" ] && args+=("seed$S=$SCRIPTS/$npz")
    done
    echo "----- $P  (n=${#args[@]}) -----" >> "$OUT"
    if [ "${#args[@]}" -eq 0 ]; then
      echo "  (no biomech dump — see eval_${P}_s*_v${VT}.log)" >> "$OUT"
    else
      python3 "$SCRIPTS/aggregate_nseed.py" "${args[@]}" >> "$OUT" 2>&1
    fi
    echo "" >> "$OUT"
  done
done
echo "ALLDONE" >> "$OUT"
cat "$OUT"
