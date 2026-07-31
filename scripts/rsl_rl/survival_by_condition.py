"""Teacher 정책의 부상 조건별 생존율.

balanced 모드(Normal/FL/FR/RL/RR 균등)로 굴리며 조건별 에피소드 길이와 조기 종료율을
집계합니다. 특정 다리 조건만 즉사하면(= mode collapse) 그 조건의 평균 길이가 급락하고
조기 종료율이 100% 에 붙습니다.

Phase 2 가 끝나면 Phase 3 로 넘어가기 전에 반드시 확인하세요. 전체 평균 에피소드
길이만 보면 "절반은 성공" 처럼 보이지만, 실제로는 한 조건만 학습되고 나머지 셋이
0.7초 만에 넘어지는 상태일 수 있습니다 (실측 사례: Normal 997 / RL 779 / FL 43 /
FR 37 / RR 33 스텝). 그런 teacher 를 증류하면 student 도 그 한 조건만 흉내 냅니다.

Usage (학습과 동일한 GO1_* 물리 설정을 그대로 준 채로):
  GO1_PHASE=teacher GO1_EVAL_MODE=balanced GO1_USE_PEG_LEG_CURRICULUM=0 ... \
  python3 survival_by_condition.py --checkpoint <phase2 model_N.pt> --headless
"""

import argparse
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip
from peg_leg_action_wrapper import PegLegActionMaskWrapper  # isort: skip

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=400)
parser.add_argument("--steps", type=int, default=1200)
parser.add_argument("--task", type=str, default="Template-Go1-Lab-v0")
parser.add_argument("--agent", type=str, default="rsl_rl_teacher_mlp_cfg_entry_point")
parser.add_argument("--seed", type=int, default=7)
parser.add_argument(
    "--full_episode_steps",
    type=int,
    default=990,
    help="이 값 이상 버틴 에피소드는 완주로 간주 (기본 episode_length 1000 기준).",
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config
import go1_lab.tasks  # noqa: F401

CONDITION_NAMES = ["Normal", "FL", "FR", "RL", "RR"]


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = PegLegActionMaskWrapper(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # RSL-RL 3.0.1+ 호환 patch (train.py/play.py 와 동일)
    d = agent_cfg.to_dict()
    for component in ["actor", "critic", "student", "teacher"]:
        if isinstance(d.get("policy", {}).get(component), dict):
            d["policy"][component].setdefault("class_name", "MLP")
    for key in ["optimizer", "config_class", "share_cnn_encoders"]:
        d.get("algorithm", {}).pop(key, None)

    runner = OnPolicyRunner(env, d, log_dir=None, device=agent_cfg.device)
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    base = env.unwrapped
    obs = env.get_observations()

    # 조건별 [완료 길이 합, 완료 에피소드 수, 조기 종료 수]
    stats = {i: [0, 0, 0] for i in range(5)}
    # done 이 관측되는 시점에는 이미 리셋되어 조건/길이가 새 값이므로 직전 값을 보관
    prev_len = base.episode_length_buf.clone()
    prev_cond = (base._peg_leg_index + 1).clone()

    with torch.inference_mode():
        for step in range(args_cli.steps):
            obs, _, dones, _ = env.step(policy(obs))
            for e in torch.where(dones)[0].tolist():
                cond = int(prev_cond[e])
                length = int(prev_len[e]) + 1
                stats[cond][0] += length
                stats[cond][1] += 1
                if length < args_cli.full_episode_steps:
                    stats[cond][2] += 1
            prev_len = base.episode_length_buf.clone()
            prev_cond = (base._peg_leg_index + 1).clone()
            if step % 300 == 0:
                print(f"  step {step}/{args_cli.steps}", flush=True)

    print(f"\n{'조건':>8} {'완료수':>7} {'평균길이':>9} {'조기종료율':>10}")
    for i in range(5):
        total, count, early = stats[i]
        if count:
            print(
                f"{CONDITION_NAMES[i]:>8} {count:>7} {total / count:>9.1f} "
                f"{100 * early / count:>9.1f}%"
            )
        else:
            print(f"{CONDITION_NAMES[i]:>8} {'-':>7} {'-':>9} {'-':>10}")
    print(
        "\n해석: 특정 조건의 조기종료율이 100% 에 가까우면 그 부상 조건은 학습되지 "
        "않은 것입니다. 완료수가 유독 큰 조건은 계속 넘어져 리셋을 반복한다는 뜻입니다."
    )

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
