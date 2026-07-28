# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlDistillationAlgorithmCfg,
    RslRlDistillationRunnerCfg,
    RslRlDistillationStudentTeacherRecurrentCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoActorCriticRecurrentCfg,
    RslRlPpoAlgorithmCfg,
)


# =====================================================================
# Phase 1: Healthy pretrain (정상 보행 선학습)
# =====================================================================

@configclass
class HealthyPPOLstmRunnerCfg(RslRlOnPolicyRunnerCfg):
    """Phase 1: 정상 로봇 보행 pretrain (PPO+LSTM).

    ⭐ 설계 원칙:
      1. **Teacher(Phase 2) 와 완전히 동일한 policy 아키텍처 + obs dim** 을 사용하여
         Phase 2 가 Phase 1 체크포인트를 `--resume` 으로 warm-start 할 수 있도록 합니다.
         → `obs_groups` 에 `privileged_obs` 를 포함 (Phase 1 에서는 [0, 0, 1.0] 로 고정).
      2. **Isaac Lab 표준 Go1 rough PPO 하이퍼파라미터** 를 최대한 반영
         (`learning_rate=1e-3`, `num_steps_per_env=24`, entropy/clip/gamma 동일).
      3. 5 frame observation stacking 은 LSTM 과 중복 메모리이므로 비사용
         (env_cfg 에서 history_length 오버라이드 제거함).
    """

    num_steps_per_env = 24
    # LSTM 은 MLP 대비 수렴이 2-3× 느림. 표준 Go1 rough(MLP 1500) 와 "동등 수준"
    # baseline 을 확보하려면 약 5-6k iter 가 필요. 3k 는 trot 기본 구조는 나오지만
    # duty factor 편차/CoM sway 가 완전히 수렴하지 않는 경우가 있어 6000 로 상향.
    max_iterations = 6000
    save_interval = 100
    experiment_name = "unitree_go1_rough_healthy"
    check_for_nan = True

    # Teacher 와 동일한 obs_groups — Phase 2 에서 resume 시 dim mismatch 없음.
    obs_groups = {
        "policy": ["policy", "privileged_obs"],
        "critic": ["policy", "privileged_obs"],
    }

    policy = RslRlPpoActorCriticRecurrentCfg(
        init_noise_std=1.0,
        noise_std_type="log",
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        rnn_type="lstm",
        rnn_hidden_dim=256,
        rnn_num_layers=1,
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        # Isaac Lab 표준 Go1 rough 와 동일 (LSTM 도 adaptive schedule 하에서
        # 1e-3 시작 → 자동으로 줄어들며 안정). 3e-4 는 지나치게 보수적이었음.
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class PPORunnerCfg(HealthyPPOLstmRunnerCfg):
    """기본 엔트리포인트: Phase 1 정상 보행 PPO+LSTM."""

    pass


# =====================================================================
# Phase 2: Teacher (peg-leg 환경 + privileged obs)
# =====================================================================

@configclass
class TeacherRunnerCfg(RslRlOnPolicyRunnerCfg):
    """Phase 2: Teacher PPO+LSTM (privileged obs 포함).

    Teacher 는 privileged_obs(부상 다리, 부목 길이, 마찰) 를 직접 관측해 최적 보행을 학습합니다.

    ⭐ **Phase 1 warm-start 구조**:
      - `HealthyPPOLstmRunnerCfg` 와 동일한 아키텍처/obs dim 이므로 Phase 1 체크포인트를
        `--resume --load_run healthy_vX --checkpoint model_N.pt` 로 이어서 학습 가능.
      - Phase 1: 정상 env 만 (peg-leg 없음) → 기본 보행 학습.
      - Phase 2: peg-leg 시나리오 50% 포함 → 기본 보행 유지 + 부상 적응 추가 학습.
    """

    num_steps_per_env = 24
    # Phase 1 에서 이미 보행이 학습되었으므로 Teacher 는 peg-leg 적응에만 집중.
    max_iterations = 5000
    save_interval = 50
    experiment_name = "unitree_go1_rough_teacher"
    check_for_nan = True

    obs_groups = {
        "policy": ["policy", "privileged_obs"],
        "critic": ["policy", "privileged_obs"],
    }

    policy = RslRlPpoActorCriticRecurrentCfg(
        init_noise_std=1.0,
        noise_std_type="log",
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        rnn_type="lstm",
        rnn_hidden_dim=256,
        rnn_num_layers=1,
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class TeacherMlpRunnerCfg(RslRlOnPolicyRunnerCfg):
    """Phase 2 antalgic teacher — feedforward MLP, NO mirror augmentation.

    In canonical RMA the teacher is FEEDFORWARD (it observes the privileged state
    directly, so needs no memory); recurrence belongs to the student adaptation
    module. The deployed student (Phase 3) stays recurrent (LSTM).

    Phase 1 (launch_phase1.sh) uses this same runner cfg with the healthy env, so
    the Phase-2 warm-start is dimension/architecture compatible. Mirror data
    augmentation is not used: under active load-bearing it drives the policy to
    the symmetric non-use optimum instead of the antalgic partial loading.
    Left/right consistency for deployment is restored by canonicalization (mirror
    the obs/action for right-side injuries at inference), and for the paper by
    the n-seed aggregate.
    """

    num_steps_per_env = 24
    max_iterations = 5000
    save_interval = 50
    experiment_name = "unitree_go1_rough_teacher"
    run_name = "phase2_mlp_teacher_plain"
    check_for_nan = True

    obs_groups = {
        "policy": ["policy", "privileged_obs"],
        "critic": ["policy", "privileged_obs"],
    }

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


# =====================================================================
# Phase 3: Student Distillation (Teacher latent 모사)
# =====================================================================

@configclass
class DistillRunnerCfg(RslRlDistillationRunnerCfg):
    """Phase 3: Student distillation.

    Teacher(Phase 2)를 동결하고 Student LSTM이
    proprioceptive history만으로 Teacher의 latent z_t를 추정합니다.

    loss: ||z_t - z_hat_t||² (MSE)
    """

    num_steps_per_env = 32
    # Phase 2 teacher 가 15000 iter 로 학습되었을 때, student 가 교사의 latent 를
    # 5가지 시나리오(정상 + FL/FR/RL/RR 부상) 전체에 걸쳐 모사하기 위해 권장 12000 iter.
    # 근거:
    #   - sample-equivalent: Phase 2(15k × 24 × 4096) = 1.47B → Phase 3 에서 동일 경험 확보에
    #     약 11250 iter (32 × 4096) 필요. 12000 은 그에 소폭 여유 추가.
    #   - 다중 시나리오 LSTM distillation 은 60-80% of teacher time 이 경험적 sweet spot.
    #   - Loss plateau 에 도달하면 save_interval 로 저장된 중간 체크포인트에서 조기 중단 가능.
    max_iterations = 12000
    save_interval = 100
    experiment_name = "unitree_go1_rough_student"
    check_for_nan = True

    obs_groups = {
        "student": ["policy"],
        "teacher": ["policy", "privileged_obs"],
    }

    policy = RslRlDistillationStudentTeacherRecurrentCfg(
        init_noise_std=0.05,
        noise_std_type="log",
        student_obs_normalization=False,
        teacher_obs_normalization=False,
        student_hidden_dims=[512, 256, 128],
        teacher_hidden_dims=[512, 256, 128],
        activation="elu",
        rnn_type="lstm",
        rnn_hidden_dim=256,
        rnn_num_layers=1,
        # The Phase-2 teacher is a FEEDFORWARD MLP (TeacherMlp pivot), so the
        # distillation must load it as non-recurrent (recurrence lives only in the
        # student LSTM). teacher_recurrent=True would expect an RNN the MLP teacher
        # checkpoint does not have.
        teacher_recurrent=False,
    )

    algorithm = RslRlDistillationAlgorithmCfg(
        num_learning_epochs=5,
        # 1e-3는 LSTM distillation에서 불안정/편향(한쪽 다리만 잘 배우는 현상)을
        # 유발할 수 있음. 5e-4로 낮춰 수렴을 안정화.
        learning_rate=5.0e-4,
        # BPTT 윈도우: 20→32로 늘려 보행 주기(약 20~30 step) + 부상 적응 long-horizon 패턴을
        # Student가 모사할 수 있도록 함. num_steps_per_env=32와 정렬.
        gradient_length=32,
        max_grad_norm=1.0,
        optimizer="adam",
        loss_type="mse",
    )
