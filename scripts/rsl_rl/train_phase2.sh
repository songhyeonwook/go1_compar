#!/bin/bash
# =============================================================================
# Phase 1 / 2 / 3 학습 — 유일한 진입점 (train.py 래퍼).
# =============================================================================
# 이름과 달리 이 스크립트는 세 phase 를 모두 담당합니다. GO1_PHASE 로 분기:
#   teacher (기본) : Phase 1(healthy) / Phase 2(antalgic) teacher-MLP 학습.
#                    Phase 1 은 GO1_PROB_PEG_LEG=0, GO1_PAIN_WEIGHT=0,
#                    GO1_USE_PEG_LEG_CURRICULUM=0 인 특수 케이스일 뿐입니다
#                    (baselines/launch_phase1.sh).
#   student        : Phase 3 distillation. 동결된 Phase 2 teacher(TEACHER_CKPT)를
#                    LSTM student 로 증류 (baselines/launch_phase3_*.sh).
# 진입점을 하나로 공유하는 것은 의도된 설계입니다: 세 phase 의 env/관측 구성이
# 어긋나는 일이 원천적으로 불가능해집니다 — Phase 2 warm-start 와 Phase 3
# distillation 모두, teacher 가 학습된 것과 정확히 같은 환경을 요구합니다.
# phase 별 차이는 (a) run identity, (b) checkpoint 로딩, (c) train.py 호출뿐이고
# 아래 모든 GO1_* env 설정은 세 phase 가 공유합니다.
#
# 이 스크립트에는 학습 로직이 없습니다 — warm-start 경로를 결정하고,
# go1_lab_env_cfg.py 가 읽어 보상/부상/커리큘럼을 조립하는 GO1_* 환경변수를
# 설정한 뒤 train.py 를 호출할 뿐입니다. 사실상 bash 로 작성된 설정 파일입니다.
#
# 아래의 모든 knob 은 `${VAR:-default}` 형태라, baselines/ 의 런처가 export 만
# 하면 오버라이드됩니다. go1_lab_env_cfg.py 기본값을 단순 반복하는 knob 은
# 생략하되, 값 자체가 핵심 실험 선택(부상 모델, 커리큘럼, eq.3 가중치,
# 3-패러다임 비교 변수)을 문서화하는 경우는 중복이어도 남겨둡니다.
# 여기의 기본값이 곧 논문 구성입니다:
#
#   reward (eq.3)  r = W_task r_task(v,v*) - W_energy ||tau||^2 - W_pain C_pain(Fz)
#   pain   (eq.4)  C_pain(Fz) = Pbase 1[contact] + max(0, exp(a (Fz - Fth)) - 1)
#                  Pbase = 0.05, Fth = 10 N, a = 2.0   (고정, severity 스케일링 없음)
#
# 처방적 gait 항들(contact/duty/diagonal asymmetry, trot_sync, front-rear
# load, intact overload)은 전부 기본 0 입니다: 논문의 주장은 antalgic 패턴이
# eq.3 만으로 창발한다는 것이므로, 어떤 항도 목표 gait/duty factor/비대칭을
# 처방해서는 안 됩니다. severity 는 통증 함수 스케일링이 아니라 형태학
# (부목의 eq.1-2 Jacobian)으로 처리합니다.
#
# 유일한 예외는 load-bearing viability floor (GO1_INJURED_*_NONUSE_*) 입니다:
# 부상 다리의 완전 비사용(non-use)을 막아, 패러다임 비교가 사용-vs-비사용이
# 아니라 gait 패턴 차원에서 이루어지게 합니다. 모든 패러다임에 동일하게
# 적용되며 — 이 선택은 논문에 명시할 것.
#
# Balance shaping (flat_orientation + 완화된 base height) 이 있는 이유:
# 짧고 단단한 peg 는 그대로 두면 몸통을 부상 쪽 모서리로 굴려(ROLLING) 땅에
# 닿으려다 넘어집니다(bad_orientation). 몸통 기울기 각도를 페널티하고 약간의
# 스쿼트를 허용하면, 수평 자세를 유지한 채 peg 에 하중을 실을 수 있게 됩니다.
#
# 사용법: 항상 baselines/launch_*.sh 를 통해 실행합니다. 런처가 실험별
# 오버라이드(PD 게인, 명령 속도 범위, 부목 기하, run name)를 공급합니다.
# 직접 실행:
#   GO1_NO_WARMSTART=1 PHASE2_RUN_NAME=my_run ./train_phase2.sh   # 처음부터
#   PHASE1_CKPT=/path/model_5999.pt PHASE2_RUN_NAME=my_run ./train_phase2.sh
# =============================================================================

set -euo pipefail
TRAIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$TRAIN_DIR"

# --- phase selection ---------------------------------------------------------
# 이 스크립트는 Phase 1/2 teacher 뿐 아니라 Phase 3 student distillation 도
# 담당합니다 (GO1_PHASE=student). env 구성을 한 곳에서 공유해야 student 가
# teacher 와 물리적으로 동일한 환경에서 distill 되기 때문입니다 — 아래 모든
# GO1_* export 는 teacher/student 가 공유하고, 분기는 (a) run identity,
# (b) checkpoint 로딩, (c) 최종 train.py 호출 뿐입니다.
IS_STUDENT=0
[ "${GO1_PHASE:-teacher}" = "student" ] && IS_STUDENT=1

# --- run identity ------------------------------------------------------------
TASK="${TASK:-Template-Go1-Lab-v0}"
if [ "$IS_STUDENT" = "1" ]; then
    EXP_NAME="${EXP_NAME:-unitree_go1_rough_student}"
    RUN_NAME="${PHASE3_RUN_NAME:-phase3_student}"
    export AGENT="${AGENT:-rsl_rl_distill_cfg_entry_point}"
    MAX_ITER="${PHASE3_MAX_ITER:-12000}"
else
    EXP_NAME="${EXP_NAME:-unitree_go1_rough_teacher}"
    RUN_NAME="${PHASE2_RUN_NAME:-phase2_teacher}"
    export AGENT="${AGENT:-rsl_rl_teacher_mlp_cfg_entry_point}"
    MAX_ITER="${PHASE2_MAX_ITER:-6000}"
fi
NUM_ENVS="${NUM_ENVS:-4096}"
SEED="${SEED:-42}"

# --- checkpoint loading ------------------------------------------------------
# Phase 2 teacher: warm-start weights from a Phase 1 checkpoint (WS_ARGS).
# Phase 3 student: load the FROZEN Phase 2 teacher into the distillation runner's
#   teacher submodule (TEACHER_CKPT). The student (LSTM) is randomly initialised
#   and learns to reproduce the teacher's latent from proprioception via MSE;
#   --warmstart_ckpt_path is ignored for distillation, so it is not passed here.
WS_ARGS=()
if [ "$IS_STUDENT" = "1" ]; then
    TEACHER_CKPT="${TEACHER_CKPT:-}"
    if [ -z "$TEACHER_CKPT" ]; then
        echo "ERROR: student distillation needs the Phase 2 teacher checkpoint."
        echo "       set TEACHER_CKPT=/path/to/model_N.pt (baselines/launch_phase3_*.sh resolves it)."
        exit 1
    elif [ ! -f "$TEACHER_CKPT" ]; then
        echo "ERROR: teacher checkpoint not found: $TEACHER_CKPT"
        exit 1
    fi
else
    # GO1_NO_WARMSTART=1 trains from scratch; this is how Phase 1 is trained, since
    # there is nothing to inherit from yet. Every Phase 2 run instead warm-starts,
    # so PHASE1_CKPT must point at the Phase 1 checkpoint
    # (baselines/launch_warmstart_*.sh resolve it, defaulting to models/).
    PHASE1_CKPT="${PHASE1_CKPT:-}"
    if [ "${GO1_NO_WARMSTART:-0}" = "1" ]; then
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
fi

# --- environment / observation ----------------------------------------------
# Proprioception-only R^48 on flat terrain (paper §4.3: "no exteroceptive
# sensing"); privileged obs stay on for the teacher. Phase 1 must use the same
# flags or the warm-start is dimension-incompatible.
export GO1_PHASE="${GO1_PHASE:-teacher}"
export GO1_PROPRIO_ONLY="${GO1_PROPRIO_ONLY:-1}"
export GO1_FLAT_TERRAIN="${GO1_FLAT_TERRAIN:-1}"
export GO1_DOMAIN_RAND="${GO1_DOMAIN_RAND:-1}"
export GO1_PHASE2_GAIT_TUNING="${GO1_PHASE2_GAIT_TUNING:-0}"
export GO1_ABS_JOINT_OBS="${GO1_ABS_JOINT_OBS:-1}"

# --- injury model (functional splint) + curriculum ---------------------------
export GO1_PEG_WEAKEN_JOINTS="${GO1_PEG_WEAKEN_JOINTS:-hip}"
# 조건(Normal/FL/FR/RL/RR)을 env id 에 고정해 조건별 학습 스텝 수를 균등하게 유지.
# "random" 은 리셋마다 재추첨 -> 학습량이 에피소드 길이에 비례 -> 빨리 넘어지는
# 조건일수록 데이터가 줄어드는 악순환 (실측 FL 43스텝 / RL 779스텝, 앞다리 전멸).
export GO1_TARGET_LEG="${GO1_TARGET_LEG:-env_fixed}"
# env_fixed 의 정상 슬롯 수. 주기 = 이 값 + 4(다리). 기본 4 -> 정상 50% / 각 다리 12.5%.
# env_fixed 는 prob_peg_leg 를 무시해 '부상 확률 커리큘럼' 이 작동하지 않으므로, 초기
# 난이도는 이 값으로 잡습니다. 1(부상 80%)로 두면 정책이 정점 후 붕괴했습니다.
export GO1_ENV_FIXED_HEALTHY_SLOTS="${GO1_ENV_FIXED_HEALTHY_SLOTS:-4}"
# 커리큘럼 램프는 ITERATION 단위인데 common_step_counter 는 정책 스텝을 셉니다.
# num_steps_per_env 로 환산해야 램프가 의도대로 됩니다 (rsl_rl_ppo_cfg.py:
# TeacherMlpRunnerCfg=24, DistillRunnerCfg=32).
if [ "$IS_STUDENT" = "1" ]; then
    export GO1_CURRICULUM_STEPS_PER_ITER="${GO1_CURRICULUM_STEPS_PER_ITER:-32}"
else
    export GO1_CURRICULUM_STEPS_PER_ITER="${GO1_CURRICULUM_STEPS_PER_ITER:-24}"
fi
export GO1_PROB_PEG_LEG="${GO1_PROB_PEG_LEG:-0.5}"
export GO1_SPLINT_LENGTH_MIN="${GO1_SPLINT_LENGTH_MIN:-0.20}"
export GO1_SPLINT_LENGTH_MAX="${GO1_SPLINT_LENGTH_MAX:-0.30}"
export GO1_FOOT_FRICTION_MIN="${GO1_FOOT_FRICTION_MIN:-0.5}"
export GO1_FOOT_FRICTION_MAX="${GO1_FOOT_FRICTION_MAX:-1.5}"

export GO1_INJURED_FOOT_FRICTION_ONLY="${GO1_INJURED_FOOT_FRICTION_ONLY:-1}"
# Splint knee PD gain. Measured (128 envs) corr(splint_length, ACTUAL PhysX angle):
#   Kp=12 -> 0.48 (sag 0.52 rad, as large as the whole 0.585 rad length range:
#                  the splint length barely reaches physics)
#   Kp=40 -> 0.95   Kp=100 -> 0.99 (sag 0.06)   Kp=300 -> 0.999 (rigid)
# 100 keeps a real splint's slight flex under load while making splint LENGTH a
# faithful experimental variable. Training and eval MUST use the same value.
export GO1_SPLINT_CALF_STIFFNESS="${GO1_SPLINT_CALF_STIFFNESS:-100}"
export GO1_SPLINT_CALF_DAMPING="${GO1_SPLINT_CALF_DAMPING:-1.0}"
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
# severity comes from MORPHOLOGY (the splint's eq.1-2 Jacobian), never from
# rescaling pain -> the severe/mild threshold+scale multipliers stay unused.
export GO1_PAIN_SEVERITY_SCALING="${GO1_PAIN_SEVERITY_SCALING:-0}"

# --- alternative impaired-limb rewards (the 3-paradigm comparison variable) ---
# antalgic = pain only; faulttol = neither; symmetry = mirror penalty only.
export GO1_SYMMETRY_PENALTY_WEIGHT="${GO1_SYMMETRY_PENALTY_WEIGHT:-0.0}"

# --- load-bearing viability floor (identical across paradigms) ----------------
export GO1_INJURED_FORCE_NONUSE_WEIGHT="${GO1_INJURED_FORCE_NONUSE_WEIGHT:--1.0}"
export GO1_INJURED_MIN_FORCE_SEVERE_N="${GO1_INJURED_MIN_FORCE_SEVERE_N:-6.0}"
export GO1_INJURED_MIN_FORCE_MILD_N="${GO1_INJURED_MIN_FORCE_MILD_N:-6.0}"
export GO1_INJURED_DUTY_NONUSE_WEIGHT="${GO1_INJURED_DUTY_NONUSE_WEIGHT:-0.0}"
export GO1_INJURED_MIN_DUTY_SEVERE="${GO1_INJURED_MIN_DUTY_SEVERE:-0.05}"
export GO1_INJURED_MIN_DUTY_MILD="${GO1_INJURED_MIN_DUTY_MILD:-0.28}"
export GO1_INJURED_NONUSE_EMA_ALPHA="${GO1_INJURED_NONUSE_EMA_ALPHA:-0.90}"
export GO1_INJURED_NONUSE_RAMP_STEPS="${GO1_INJURED_NONUSE_RAMP_STEPS:-2000}"

# --- task / energy / balance shaping -----------------------------------------
export GO1_TRACK_LIN_VEL_WEIGHT="${GO1_TRACK_LIN_VEL_WEIGHT:-2.5}"
export GO1_TORQUE_PENALTY_SCALE="${GO1_TORQUE_PENALTY_SCALE:-30}"
export GO1_SURVIVAL_BONUS_WEIGHT="${GO1_SURVIVAL_BONUS_WEIGHT:-0.5}"
export GO1_FLAT_ORIENTATION_WEIGHT="${GO1_FLAT_ORIENTATION_WEIGHT:--1.5}"
export GO1_BASE_HEIGHT_TARGET="${GO1_BASE_HEIGHT_TARGET:-0.27}"
export GO1_BASE_HEIGHT_WEIGHT="${GO1_BASE_HEIGHT_WEIGHT:--0.15}"

# --- prescritive gait terms: OFF (would contradict the emergence claim) ------
# These MUST be set: go1_lab_env_cfg.py enables every one of them by default.
# Their ramp/threshold parameters are left at the source defaults since a zero
# weight makes them inert.p
export GO1_CONTACT_FORCE_ASYM_WEIGHT="${GO1_CONTACT_FORCE_ASYM_WEIGHT:-0.0}"
export GO1_DUTY_FACTOR_ASYM_WEIGHT="${GO1_DUTY_FACTOR_ASYM_WEIGHT:-0.0}"
export GO1_DIAGONAL_LOAD_ASYM_WEIGHT="${GO1_DIAGONAL_LOAD_ASYM_WEIGHT:-0.0}"
export GO1_FRONT_REAR_LOAD_DIST_WEIGHT="${GO1_FRONT_REAR_LOAD_DIST_WEIGHT:-0.0}"
export GO1_TROT_SYNC_WEIGHT="${GO1_TROT_SYNC_WEIGHT:-0.0}"

echo "-----------------------------------------------"
if [ "$IS_STUDENT" = "1" ]; then
    echo "  Phase 3: student distillation (frozen teacher -> LSTM student)"
    echo "  teacher_ckpt=$TEACHER_CKPT"
else
    echo "  Phase 2: peg-leg teacher training"
    echo "  warmstart=$PHASE1_CKPT"
fi
echo "  run_name=$RUN_NAME  agent=$AGENT"
echo "  num_envs=$NUM_ENVS max_iterations=$MAX_ITER seed=$SEED"
echo "  peg_leg_prob=$GO1_PROB_PEG_LEG curriculum=$GO1_USE_PEG_LEG_CURRICULUM"
echo "  splint_range=[$GO1_SPLINT_LENGTH_MIN, $GO1_SPLINT_LENGTH_MAX] m"
echo "  pain: weight=$GO1_PAIN_WEIGHT threshold=${GO1_PAIN_THRESHOLD_N}N scale=$GO1_PAIN_SCALE base=$GO1_PAIN_BASE_CONTACT_COST"
echo "  symmetry_penalty=$GO1_SYMMETRY_PENALTY_WEIGHT"
echo "  viability floor: force=$GO1_INJURED_FORCE_NONUSE_WEIGHT@${GO1_INJURED_MIN_FORCE_SEVERE_N}N duty=$GO1_INJURED_DUTY_NONUSE_WEIGHT@$GO1_INJURED_MIN_DUTY_SEVERE ramp=$GO1_INJURED_NONUSE_RAMP_STEPS ema=$GO1_INJURED_NONUSE_EMA_ALPHA"
echo "  balance: flat_orientation=$GO1_FLAT_ORIENTATION_WEIGHT base_height=$GO1_BASE_HEIGHT_TARGET@$GO1_BASE_HEIGHT_WEIGHT"
echo "  gait_tuning=$GO1_PHASE2_GAIT_TUNING"
echo "  abs_joint_obs=$GO1_ABS_JOINT_OBS (1 -> policy obs 52) injured_foot_friction_only=$GO1_INJURED_FOOT_FRICTION_ONLY"
echo "  splint knee: stiffness=$GO1_SPLINT_CALF_STIFFNESS damping=$GO1_SPLINT_CALF_DAMPING"
echo "  target_leg=$GO1_TARGET_LEG (env_fixed -> 조건별 학습량 균등) healthy_slots=$GO1_ENV_FIXED_HEALTHY_SLOTS -> 부상 $(( 400 / (GO1_ENV_FIXED_HEALTHY_SLOTS + 4) ))%"
echo "  curriculum_steps_per_iter=$GO1_CURRICULUM_STEPS_PER_ITER"
echo "-----------------------------------------------"

if [ "$IS_STUDENT" = "1" ]; then
    # Distillation: load the frozen Phase 2 teacher; no warm-start.
    python3 train.py \
        --task "$TASK" \
        --agent "$AGENT" \
        --num_envs "$NUM_ENVS" \
        --headless \
        --experiment_name "$EXP_NAME" \
        --run_name "$RUN_NAME" \
        --teacher_ckpt_path "$TEACHER_CKPT" \
        --max_iterations "$MAX_ITER" \
        --seed "$SEED" \
        --use_peg_leg_action_mask
else
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
fi

echo ""
echo "-----------------------------------------------"
if [ "$IS_STUDENT" = "1" ]; then
    echo "  Phase 3 complete"
else
    echo "  Phase 2 complete"
fi
echo "  Logs: $TRAIN_DIR/logs/rsl_rl/$EXP_NAME"
echo "-----------------------------------------------"
