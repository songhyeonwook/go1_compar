"""phase3 LSTM student 롤아웃에서 (hidden, 부상 라벨) 쌍을 덤프합니다.

go1_real/scripts/train_injury_probe.py 의 입력을 만드는 시뮬 쪽 스크립트.
출력 npz: h (T,256) f16, labels (T,K) f32, names (K,) — 리셋 후 --skip-steps
스텝(LSTM 부상 추론 수렴 전)은 이미 제외되어 있으므로 probe 학습 시
--skip-steps 0 으로 쓰면 됩니다.

라벨 (K=6, deploy.py [EST] 표시 순서와 동일):
  peg_FL, peg_FR, peg_RL, peg_RR (one-hot), splint_len (m), friction

hidden 은 policy(obs_t) 호출 직후의 h_t (deploy.py numpy 백엔드의
policy.hidden 과 같은 시점) 를 기록하고, 에피소드 경계에서
policy.reset(dones) 로 hidden 을 0 리셋합니다 (학습 때와 동일).

사용 (launch_phase3_antalgic.sh 와 같은 GO1_* 물리 블록 + 균등 조건):
  GO1_EVAL_MODE=balanced ... python3 collect_student_hidden.py \
      --checkpoint <phase3 model_N.pt> --out biomech/hidden_s42.npz --headless
"""

import argparse
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip
from peg_leg_action_wrapper import PegLegActionMaskWrapper  # isort: skip

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--steps", type=int, default=1500)
parser.add_argument("--task", type=str, default="Template-Go1-Lab-v0")
parser.add_argument("--agent", type=str, default="rsl_rl_distill_cfg_entry_point")
parser.add_argument("--out", type=str, required=True)
parser.add_argument("--seed", type=int, default=7)
parser.add_argument(
    "--skip-steps", type=int, default=50,
    help="리셋 후 이 스텝 수만큼 프레임 제외 (LSTM 수렴 전, 50=1초)",
)
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

LABEL_NAMES = ["peg_FL", "peg_FR", "peg_RL", "peg_RR", "splint_len", "friction"]


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
    policy_nn = runner.alg.policy

    def student_hidden() -> torch.Tensor:
        """(E, 256) — LSTM h (student memory). num_layers=1 전제, 위반 시 에러."""
        hs = policy_nn.memory_s.hidden_states
        h = hs[0] if isinstance(hs, tuple) else hs  # LSTM: (h, c)
        if h.shape[0] != 1:
            raise RuntimeError(f"num_layers={h.shape[0]} != 1 — probe 형식 재검토 필요")
        return h[0]

    base = env.unwrapped
    num_envs = env.num_envs
    obs = env.get_observations()
    h_hist, label_hist, valid_hist = [], [], []
    # 첫 obs 도 리셋 직후이므로 카운터 0 에서 시작
    steps_since_reset = torch.zeros(num_envs, dtype=torch.long)

    with torch.inference_mode():
        for step in range(args_cli.steps):
            # 라벨은 현재 에피소드의 부상 파라미터 (env.step 의 리셋에서 갱신됨)
            idx = base._peg_leg_index.detach().cpu().numpy()  # -1=정상, 0..3=FL..RR
            one_hot = np.zeros((num_envs, 4), dtype=np.float32)
            injured = idx >= 0
            one_hot[np.arange(num_envs)[injured], idx[injured]] = 1.0
            labels = np.concatenate(
                [
                    one_hot,
                    base._peg_leg_splint_length.detach().cpu().numpy()[:, None],
                    base._peg_leg_foot_friction.detach().cpu().numpy()[:, None],
                ],
                axis=1,
            ).astype(np.float32)

            actions = policy(obs)  # 이 호출이 h_t 를 갱신
            h_hist.append(student_hidden().detach().cpu().numpy().astype(np.float16))
            label_hist.append(labels)
            valid_hist.append((steps_since_reset >= args_cli.skip_steps).numpy())

            obs, _, dones, _ = env.step(actions)
            if dones is not None:
                policy_nn.reset(dones)  # 학습과 동일: 에피소드 경계에서 hidden 0
                done_mask = dones.detach().cpu().to(torch.bool)
                steps_since_reset += 1
                steps_since_reset[done_mask] = 0
            if step % 200 == 0:
                print(f"  step {step}/{args_cli.steps}", flush=True)

    h = np.array(h_hist)          # (S, E, 256)
    labels = np.array(label_hist)  # (S, E, 6)
    valid = np.array(valid_hist)   # (S, E)
    h_flat = h[valid]              # (T, 256)
    labels_flat = labels[valid]    # (T, 6)

    # 조건별 프레임 수 리포트 — 어느 조건이 데이터 부족인지 바로 보이게
    cond = labels_flat[:, :4]
    n_normal = int((cond.sum(axis=1) == 0).sum())
    print(f"[COLLECT] {valid.size} 프레임 중 {h_flat.shape[0]} 유지 "
          f"(리셋 후 {args_cli.skip_steps} 스텝 제외)")
    print(f"  Normal={n_normal}", end="")
    for i, n in enumerate(LABEL_NAMES[:4]):
        print(f" {n}={int(cond[:, i].sum())}", end="")
    print()

    out_dir = os.path.dirname(args_cli.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(
        args_cli.out,
        h=h_flat,
        labels=labels_flat,
        names=np.array(LABEL_NAMES),
    )
    print(f"[SAVED] {args_cli.out}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
