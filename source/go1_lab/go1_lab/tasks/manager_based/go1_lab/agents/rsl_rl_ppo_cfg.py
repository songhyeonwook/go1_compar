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
    RslRlPpoAlgorithmCfg,
)


# =====================================================================
# Phase 1/2: Teacher (MLP, privileged obs)
# =====================================================================

@configclass
class TeacherMlpRunnerCfg(RslRlOnPolicyRunnerCfg):
    """Phase 1/2 antalgic teacher — feedforward MLP, mirror augmentation 미사용.

    정통 RMA 에서 teacher 는 FEEDFORWARD 입니다 (privileged state 를 직접
    관측하므로 메모리가 불필요). 시계열 재귀(recurrence)는 student 의
    adaptation module 몫이며, 배포되는 student(Phase 3)는 LSTM 을 유지합니다.

    Phase 1(launch_phase1.sh)도 healthy env 에서 이 runner cfg 를 그대로
    사용하므로, Phase 2 warm-start 가 차원/아키텍처 호환됩니다. Mirror data
    augmentation 은 사용하지 않습니다: 능동적 하중 부하(load-bearing) 하에서는
    정책을 antalgic 부분 하중이 아닌 좌우 대칭 비사용(non-use) 최적해로
    몰아가기 때문입니다. 배포 시 좌우 일관성은 canonicalization(오른쪽 부상이면
    추론 시 obs/action 미러링)으로, 논문에서는 n-seed 집계로 확보합니다.
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
        teacher_recurrent=False,
    )

    algorithm = RslRlDistillationAlgorithmCfg(
        num_learning_epochs=5,
        learning_rate=5.0e-4,
        gradient_length=32,
        max_grad_norm=1.0,
        optimizer="adam",
        loss_type="mse",
    )
