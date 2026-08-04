from __future__ import annotations

import os
import torch
from typing import TYPE_CHECKING

from isaaclab.envs.utils.io_descriptors import (
    generic_io_descriptor,
    record_dtype,
    record_shape,
)

from .events import CALF_JOINT_NAMES

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# generic_io_descriptor: env.export_IO_descriptors() 가 각 obs 함수를
# inspect=True 로 호출해 IO_descriptors.yaml 에 스펙(shape/dtype/units 등)을
# 기록할 수 있게 합니다. 데코레이터가 없는 함수는 내보내기에서 조용히
# 빠지므로, 이 파일의 모든 obs 함수는 반드시 데코레이터를 달아야 합니다.


@generic_io_descriptor(
    observation_type="PegLegPrivileged",
    units="index",
    on_inspect=[record_shape, record_dtype],
)
def peg_leg_index(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """부상 상태 인덱스: 0=정상, 1=FL, 2=FR, 3=RL, 4=RR."""
    if os.getenv("GO1_HIDE_PRIVILEGED_INJURY", "0").strip().lower() in {"1", "true", "yes", "on"}:
        return torch.zeros((env.num_envs, 1), device=env.device)
    if hasattr(env, "_peg_leg_index"):
        return (env._peg_leg_index.float() + 1.0).unsqueeze(-1)
    return torch.zeros((env.num_envs, 1), device=env.device)


@generic_io_descriptor(
    observation_type="PegLegPrivileged",
    units="one_hot",
    element_order=["FL", "FR", "RL", "RR", "injured_flag"],
    on_inspect=[record_shape, record_dtype],
)
def peg_leg_one_hot(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """고장 다리 one-hot (FL, FR, RL, RR)와 부상 플래그를 반환합니다."""
    one_hot = torch.zeros((env.num_envs, 5), device=env.device)
    if hasattr(env, "_peg_leg_index"):
        idx = env._peg_leg_index.to(dtype=torch.long)
        valid = idx >= 0
        if torch.any(valid):
            one_hot[valid, idx[valid]] = 1.0
            one_hot[valid, 4] = 1.0  # injured flag
    return one_hot


@generic_io_descriptor(
    observation_type="PegLegPrivileged",
    units="dimensionless",
    on_inspect=[record_shape, record_dtype],
)
def peg_leg_foot_friction(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """고장 다리 부목 발 마찰 계수를 반환합니다 (정상 = 0, 부목 없음)."""
    if os.getenv("GO1_HIDE_PRIVILEGED_INJURY", "0").strip().lower() in {"1", "true", "yes", "on"}:
        return torch.zeros((env.num_envs, 1), device=env.device)
    if hasattr(env, "_peg_leg_foot_friction"):
        return env._peg_leg_foot_friction.unsqueeze(-1)
    return torch.zeros((env.num_envs, 1), device=env.device)


@generic_io_descriptor(
    observation_type="PegLegPrivileged",
    units="m",
    on_inspect=[record_shape, record_dtype],
)
def peg_leg_splint_length(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """부목 등가 길이(m)를 반환합니다. Go1 링크 기구학 기반."""
    if os.getenv("GO1_HIDE_PRIVILEGED_INJURY", "0").strip().lower() in {"1", "true", "yes", "on"}:
        return torch.zeros((env.num_envs, 1), device=env.device)
    if hasattr(env, "_peg_leg_splint_length"):
        return env._peg_leg_splint_length.unsqueeze(-1)
    return torch.zeros((env.num_envs, 1), device=env.device)


@generic_io_descriptor(
    observation_type="JointState",
    units="rad",
    joint_names=CALF_JOINT_NAMES,
    on_inspect=[record_shape, record_dtype],
)
def calf_pos_nominal_rel(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """calf 관절각 − 부상 전 nominal (FL, FR, RL, RR 순).

    joint_pos_rel 과 달리 부상 시 재작성되는 default 가 아니라 부상 전
    nominal 기준이므로, 부목 lock 각이 관측에서 소거되지 않습니다
    (GO1_ABS_JOINT_OBS=1 로 policy 그룹에 추가됨).
    """
    asset = env.scene["robot"]
    joint_names = list(asset.data.joint_names)
    calf_ids = [joint_names.index(n) for n in CALF_JOINT_NAMES if n in joint_names]
    if len(calf_ids) != len(CALF_JOINT_NAMES):
        return torch.zeros((env.num_envs, len(CALF_JOINT_NAMES)), device=env.device)

    nominal = getattr(env, "_peg_leg_default_joint_pos_ref", None)
    if nominal is None:
        # 첫 리셋 이전(= 아직 아무 관절도 lock 되지 않음)에는 현재 default 가 곧 nominal.
        default = asset.data.default_joint_pos
        nominal = default[0] if default.ndim == 2 else default
    nominal = nominal.to(env.device)

    return asset.data.joint_pos[:, calf_ids] - nominal[calf_ids].unsqueeze(0)
