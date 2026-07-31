"""Roll out a student policy and record (proprio obs history, gt injury params).

Feasibility data for supervised injury-parameter estimation: can the injured
leg / splint length / foot friction be decoded from the 48-dim proprio history?
Pair with train_injury_estimator.py (offline, no sim).

Run from scripts/rsl_rl with the same GO1_* env block as the phase3 launcher;
GO1_CMD_VX_MIN/MAX fix the commanded speed (set both equal for a fixed speed).

Output npz: obs (S,E,48) f16, idx (S,E) i8, splint (S,E) f32, fric (S,E) f32.
"""

import argparse
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip
from peg_leg_action_wrapper import PegLegActionMaskWrapper  # isort: skip

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--steps", type=int, default=1500)
parser.add_argument("--task", type=str, default="Template-Go1-Lab-v0")
parser.add_argument("--agent", type=str, default="rsl_rl_distill_cfg_entry_point")
parser.add_argument("--out", type=str, required=True)
parser.add_argument("--seed", type=int, default=7)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import os

import numpy as np
import torch

from rsl_rl.runners import DistillationRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config
import go1_lab.tasks  # noqa: F401


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = PegLegActionMaskWrapper(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # RSL-RL 3.0.1+ 호환 patch (train.py/play.py와 동일)
    agent_cfg_dict = agent_cfg.to_dict()
    for component in ["actor", "critic", "student", "teacher"]:
        c = agent_cfg_dict.get("policy", {}).get(component)
        if isinstance(c, dict):
            c.setdefault("class_name", "MLP")
    for k in ["optimizer", "config_class", "share_cnn_encoders"]:
        agent_cfg_dict.get("algorithm", {}).pop(k, None)

    runner = DistillationRunner(env, agent_cfg_dict, log_dir=None, device=agent_cfg.device)
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    base = env.unwrapped
    obs = env.get_observations()
    obs_hist, idx_hist, spl_hist, fric_hist = [], [], [], []
    with torch.inference_mode():
        for step in range(args_cli.steps):
            obs_hist.append(obs["policy"].detach().cpu().numpy().astype(np.float16))
            idx_hist.append(base._peg_leg_index.detach().cpu().numpy().astype(np.int8))
            spl_hist.append(base._peg_leg_splint_length.detach().cpu().numpy().astype(np.float32))
            fric_hist.append(base._peg_leg_foot_friction.detach().cpu().numpy().astype(np.float32))
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)
            if step % 200 == 0:
                print(f"  step {step}/{args_cli.steps}", flush=True)

    out_dir = os.path.dirname(args_cli.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(
        args_cli.out,
        obs=np.array(obs_hist),
        idx=np.array(idx_hist),
        splint=np.array(spl_hist),
        fric=np.array(fric_hist),
    )
    print(f"[SAVED] {args_cli.out}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
