# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##


gym.register(
    id="Template-Go1-Lab-v0",
    entry_point=f"{__name__}.go1_lab_env:Go1LabEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go1_lab_env_cfg:Go1LabEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
        "rsl_rl_teacher_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TeacherRunnerCfg",
        "rsl_rl_teacher_mlp_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TeacherMlpRunnerCfg",
        "rsl_rl_distill_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DistillRunnerCfg",
    },
)
