#!/usr/bin/env python3
"""Export a selected RSL-RL Go1 policy checkpoint to JIT and ONNX, then exit."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

import cli_args  # isort: skip
from peg_leg_action_wrapper import PegLegActionMaskWrapper  # isort: skip


parser = argparse.ArgumentParser(description="Export an RSL-RL Go1 checkpoint to JIT/ONNX.")
# NOTE: --checkpoint is provided by cli_args.add_rsl_rl_args (same dest); do not redefine it here (argparse conflict).
parser.add_argument("--output_dir", type=str, default=None, help="Export directory. Default: <checkpoint_dir>/exported.")
parser.add_argument("--task", type=str, default="Template-Go1-Lab-v0")
parser.add_argument("--agent", type=str, default="rsl_rl_distill_cfg_entry_point")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--phase", type=str, default="student", choices=("healthy", "teacher", "student"))
parser.add_argument("--jit_name", type=str, default="policy.pt")
parser.add_argument("--onnx_name", type=str, default="policy.onnx")
parser.add_argument("--descriptor_name", type=str, default="policy_io.json")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
os.environ["GO1_PHASE"] = args_cli.phase
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

import go1_lab.tasks  # noqa: F401


def _selected_phase3_checkpoint() -> str:
    path = Path(__file__).resolve().parent / "logs" / "rsl_rl" / "unitree_go1_rough_student"
    selected = path / "PAPER_GRADE_PHASE3_CHECKPOINT.txt"
    if selected.is_file():
        value = selected.read_text(encoding="utf-8").strip()
        if value and value != "NO_PAPER_GRADE_CANDIDATE":
            return value
    raise FileNotFoundError(
        "No selected Phase 3 checkpoint. Pass --checkpoint or run select_phase3_student_candidate.py first."
    )


def _normalizer(policy_nn):
    if hasattr(policy_nn, "actor_obs_normalizer"):
        return policy_nn.actor_obs_normalizer
    if hasattr(policy_nn, "student_obs_normalizer"):
        return policy_nn.student_obs_normalizer
    return None


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    checkpoint = os.path.abspath(args_cli.checkpoint or _selected_phase3_checkpoint())
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.log_dir = os.path.dirname(checkpoint)

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = PegLegActionMaskWrapper(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    agent_cfg_dict = agent_cfg.to_dict()

    # RSL-RL 3.0.1+ 버전 호환성을 위한 Patch (train.py/play.py와 동일해야 체크포인트가 로드됨)
    # 1. policy 구성 요소에 class_name 주입
    if "policy" in agent_cfg_dict:
        policy_cfg = agent_cfg_dict["policy"]
        for component in ["actor", "critic", "student", "teacher"]:
            if component in policy_cfg and isinstance(policy_cfg[component], dict):
                if "class_name" not in policy_cfg[component]:
                    policy_cfg[component]["class_name"] = "MLP"

    # 2. algorithm 설정에서 PPO 클래스가 지원하지 않는 키워드 제거
    if "algorithm" in agent_cfg_dict:
        alg_cfg = agent_cfg_dict["algorithm"]
        for taboo_key in ["optimizer", "config_class", "share_cnn_encoders"]:
            if taboo_key in alg_cfg:
                alg_cfg.pop(taboo_key)

    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg_dict, log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg_dict, log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")

    print(f"[INFO] Loading checkpoint: {checkpoint}")
    runner.load(checkpoint)
    policy_nn = runner.alg.policy if hasattr(runner.alg, "policy") else runner.alg.actor_critic
    normalizer = _normalizer(policy_nn)

    export_dir = Path(args_cli.output_dir or (Path(checkpoint).parent / "exported"))
    export_dir.mkdir(parents=True, exist_ok=True)
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=str(export_dir), filename=args_cli.jit_name)
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=str(export_dir), filename=args_cli.onnx_name)

    obs = env.get_observations()
    obs_policy = obs["policy"] if isinstance(obs, dict) else obs
    policy_is_recurrent = bool(getattr(policy_nn, "is_recurrent", False))
    descriptor = {
        "checkpoint": checkpoint,
        "phase": args_cli.phase,
        "agent": args_cli.agent,
        "task": args_cli.task,
        "jit": str(export_dir / args_cli.jit_name),
        "onnx": str(export_dir / args_cli.onnx_name),
        "is_recurrent": policy_is_recurrent,
        "observation_shape": list(obs_policy.shape),
        "action_shape": list(env.action_space.shape),
    }
    if policy_is_recurrent:
        rnn = policy_nn.memory_s.rnn if hasattr(policy_nn, "memory_s") else policy_nn.memory_a.rnn
        descriptor.update(
            {
                "rnn_type": type(rnn).__name__.lower(),
                "rnn_num_layers": int(rnn.num_layers),
                "rnn_hidden_size": int(rnn.hidden_size),
                "onnx_inputs": ["obs", "h_in", "c_in"] if type(rnn).__name__.lower() == "lstm" else ["obs", "h_in"],
                "onnx_outputs": ["actions", "h_out", "c_out"]
                if type(rnn).__name__.lower() == "lstm"
                else ["actions", "h_out"],
            }
        )
    else:
        descriptor.update({"onnx_inputs": ["obs"], "onnx_outputs": ["actions"]})

    descriptor_path = export_dir / args_cli.descriptor_name
    descriptor_path.write_text(json.dumps(descriptor, indent=2) + "\n", encoding="utf-8")

    io_desc_path = _export_io_descriptors(env, export_dir)

    print(f"[SAVED] {export_dir / args_cli.jit_name}")
    print(f"[SAVED] {export_dir / args_cli.onnx_name}")
    print(f"[SAVED] {descriptor_path}")
    if io_desc_path is not None:
        print(f"[SAVED] {io_desc_path}")
    env.close()


def _export_io_descriptors(env, export_dir: Path) -> Path | None:
    """Isaac Lab IO descriptor 를 IO_descriptors.yaml 로 내보냅니다.

    env.export_IO_descriptors() 는 관측을 policy 그룹만 내보내는데, teacher
    actor 는 privileged_obs 까지 연결해 받으므로 (agent.yaml obs_groups)
    두 그룹을 모두 담아야 배포 측이 59차원 레이아웃을 재구성할 수 있습니다.
    ObservationManager.get_IO_descriptors 는 그룹 인자를 받는 property 라
    fget 으로 직접 호출합니다.

    obs 함수가 inspect=True 를 지원하지 않으면 (generic_io_descriptor 미적용)
    해당 term 은 YAML 에서 빠집니다 — go1_lab/mdp/observations.py 참고.
    """
    import yaml

    try:
        from isaaclab.envs.utils.io_descriptors import (
            export_articulations_data,
            export_scene_data,
        )

        base_env = env.unwrapped
        om = base_env.observation_manager
        obs_desc = type(om).get_IO_descriptors.fget(
            om, ["policy", "privileged_obs"]
        )
        data = {
            "observations": obs_desc,
            "actions": base_env.action_manager.get_IO_descriptors,
            "articulations": export_articulations_data(base_env),
            "scene": export_scene_data(base_env),
        }
        path = export_dir / "IO_descriptors.yaml"
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
        return path
    except Exception as exc:  # noqa: BLE001 — 내보내기 실패가 export 전체를 막으면 안 됨
        print(f"[WARN] IO descriptor export failed: {type(exc).__name__}: {exc}")
        return None


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
