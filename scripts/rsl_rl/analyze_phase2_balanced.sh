#!/bin/bash
# Balanced Phase 2 teacher validation: Normal/FL/FR/RL/RR = 1:1:1:1:1.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

EXP_NAME="${EXP_NAME:-unitree_go1_rough_teacher}"
RUN_NAME="${PHASE2_RUN_NAME:-phase2_teacher_antalgic_limp_v7}"
CHECKPOINT="${CHECKPOINT:-}"
TASK="${TASK:-Template-Go1-Lab-v0}"
AGENT="${AGENT:-}"
USE_PRETRAINED_CHECKPOINT="${USE_PRETRAINED_CHECKPOINT:-0}"
PRETRAINED_TASK="${PRETRAINED_TASK:-}"
NUM_ENVS="${NUM_ENVS:-1000}"
STEPS="${STEPS:-2000}"
TARGET_VX="${TARGET_VX:-1.0}"
LOAD_CONTACT_THRESHOLD="${LOAD_CONTACT_THRESHOLD:-10.0}"
METRICS_JSON="${METRICS_JSON:-}"

if [ -z "$CHECKPOINT" ] && [ "$USE_PRETRAINED_CHECKPOINT" != "1" ]; then
    LOG_ROOT="$SCRIPT_DIR/logs/rsl_rl/$EXP_NAME"
    RUN_DIR="$(find "$LOG_ROOT" -maxdepth 1 -type d -name "*_${RUN_NAME}" | sort | tail -n 1 || true)"
    if [ -z "$RUN_DIR" ]; then
        echo "ERROR: run not found under $LOG_ROOT with suffix $RUN_NAME"
        echo "       override with CHECKPOINT=/path/to/model_N.pt"
        exit 1
    fi
    CHECKPOINT="$(
        find "$RUN_DIR" -maxdepth 1 -type f -name 'model_*.pt' \
            | awk -F'[_/.]' '{ print $(NF-1) "\t" $0 }' \
            | sort -n \
            | tail -n 1 \
            | cut -f2-
    )"
fi

if [ "$USE_PRETRAINED_CHECKPOINT" != "1" ] && [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: checkpoint not found: $CHECKPOINT"
    exit 1
fi

echo "-----------------------------------------------"
echo "  Phase 2 balanced validation"
echo "  task=$TASK"
if [ "$USE_PRETRAINED_CHECKPOINT" = "1" ]; then
    echo "  checkpoint=Isaac Lab published pretrained ($PRETRAINED_TASK)"
else
    echo "  checkpoint=$CHECKPOINT"
fi
echo "  num_envs=$NUM_ENVS steps=$STEPS vx=$TARGET_VX"
echo "  load_contact_threshold=$LOAD_CONTACT_THRESHOLD N"
echo "  splint_range=[${GO1_SPLINT_LENGTH_MIN:-0.20}, ${GO1_SPLINT_LENGTH_MAX:-0.30}] m"
echo "-----------------------------------------------"

METRICS_ARG=()
if [ -n "$METRICS_JSON" ]; then
    METRICS_ARG=(--metrics_json "$METRICS_JSON")
fi
AGENT_ARG=()
if [ -n "$AGENT" ]; then
    AGENT_ARG=(--agent "$AGENT")
fi
PRETRAINED_ARG=()
if [ "$USE_PRETRAINED_CHECKPOINT" = "1" ]; then
    PRETRAINED_ARG=(--use_pretrained_checkpoint)
    if [ -n "$PRETRAINED_TASK" ]; then
        PRETRAINED_ARG+=(--pretrained_task "$PRETRAINED_TASK")
    fi
fi

# ⚠️ 평가 env 는 학습 env 와 물리/관측이 같아야 합니다. 아래 값은 train_phase2.sh 의
# 기본값과 1:1 로 맞춘 것입니다. 특히 GO1_ABS_JOINT_OBS 는 policy 관측 차원을
# 48<->52 로 바꾸므로, 학습(1)과 평가(소스 기본 0)가 어긋나면 checkpoint 의
# strict load_state_dict 가 실패합니다 — eval_compar.sh 는 `|| true` 로 감싸고 있어
# 그 실패가 조용히 삼켜지고 결과 표가 통째로 비어버립니다.
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-go1}" \
GO1_PHASE="${GO1_PHASE:-teacher}" \
GO1_ABS_JOINT_OBS="${GO1_ABS_JOINT_OBS:-1}" \
GO1_INJURED_FOOT_FRICTION_ONLY="${GO1_INJURED_FOOT_FRICTION_ONLY:-1}" \
GO1_SPLINT_LENGTH_MIN="${GO1_SPLINT_LENGTH_MIN:-0.20}" \
GO1_SPLINT_LENGTH_MAX="${GO1_SPLINT_LENGTH_MAX:-0.30}" \
GO1_FOOT_FRICTION_MIN="${GO1_FOOT_FRICTION_MIN:-0.5}" \
GO1_FOOT_FRICTION_MAX="${GO1_FOOT_FRICTION_MAX:-1.5}" \
GO1_SPLINT_CALF_STIFFNESS="${GO1_SPLINT_CALF_STIFFNESS:-100}" \
GO1_SPLINT_CALF_DAMPING="${GO1_SPLINT_CALF_DAMPING:-1.0}" \
GO1_PEG_GRACE_STEPS="${GO1_PEG_GRACE_STEPS:-30}" \
GO1_DOMAIN_RAND="${GO1_DOMAIN_RAND:-1}" \
python3 -u analyze_student.py \
    --checkpoint "$CHECKPOINT" \
    --task "$TASK" \
    "${AGENT_ARG[@]}" \
    "${PRETRAINED_ARG[@]}" \
    --num_envs "$NUM_ENVS" \
    --steps "$STEPS" \
    --flat \
    --contact_use_z_only \
    --load_contact_threshold "$LOAD_CONTACT_THRESHOLD" \
    --target_vx "$TARGET_VX" \
    --headless \
    --seed "${SEED:-42}" \
    --balance \
    "${METRICS_ARG[@]}"
