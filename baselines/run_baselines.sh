#!/bin/bash
# Train all 3 paradigms (antalgic + faulttol + symmetry) for one seed, then
# evaluate and print the 3-paradigm comparison. Self-contained / portable.
#
# Run it detached, e.g.:   nohup ./run_baselines.sh 42 > run42.log 2>&1 &
# (or inside tmux/screen). Activate your Isaac Lab python env first.
#
# GPU/RAM note: trains up to 3 policies in parallel (set PARALLEL=1 to serialise
# on a small machine). Each teacher ~18k iters.
set -uo pipefail
S="${1:-42}"
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARALLEL="${PARALLEL:-3}"          # how many paradigms to run at once
PARADIGMS="${PARADIGMS:-antalgic faulttol symmetry}"

echo "[run_baselines] seed=$S paradigms='$PARADIGMS' parallel=$PARALLEL"
pids=()
running=0
for P in $PARADIGMS; do
  bash "$BASE_DIR/launch_compar.sh" "$P" "$S" > "$BASE_DIR/train_${P}_$S.log" 2>&1 &
  pids+=($!); running=$((running+1))
  echo "  started $P (pid $!)"
  if [ "$running" -ge "$PARALLEL" ]; then wait -n 2>/dev/null || wait; running=$((running-1)); fi
done
wait
echo "[run_baselines] all trainings done -> evaluating"
bash "$BASE_DIR/eval_compar.sh" "$S"
