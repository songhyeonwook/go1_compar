# 관측 차원 변경 정리: 48 → 59

논문 기준 proprioception-only 관측은 **48차원**이었으나, 현재 파이프라인(phase1_mlp_s42
이후)은 **59차원**으로 학습된다. 두 개의 독립적인 변경이 합쳐진 결과다:

| 구성 | 변경 전 | 변경 후 | 활성화 플래그 |
|---|---|---|---|
| policy 그룹 (proprioception) | 48 | **52** (+4, `calf_pos_nominal_rel`) | `GO1_ABS_JOINT_OBS=1` |
| privileged_obs 그룹 (teacher 전용) | 3 | **7** (+4, injury one-hot) | `GO1_INJURY_ONEHOT=1` |
| **teacher actor/critic 입력 (concat)** | 51 | **59** | — |

Phase 3 distillation 에서 student(LSTM)는 policy 그룹 **52차원만** 받고, teacher 는
59차원을 받는다 (`agents/rsl_rl_ppo_cfg.py`의 `obs_groups` 참고).

---

## 변경 1 — policy 그룹 48 → 52: `calf_pos_nominal_rel` 추가

### 어디에 추가되었나

| 코드 | 위치 | 내용 |
|---|---|---|
| 관측 함수 | `source/go1_lab/go1_lab/tasks/manager_based/go1_lab/mdp/observations.py:90` (`calf_pos_nominal_rel`) | calf 관절각 − **부상 전** nominal. `(num_envs, 4)`, 순서 [FL, FR, RL, RR] |
| policy 그룹 등록 | `go1_lab_env_cfg.py:245-246` | `GO1_ABS_JOINT_OBS=1` 이면 `calf_pos_abs` 라는 이름으로 policy 그룹 맨 뒤(인덱스 48~51)에 추가 |
| nominal 스냅샷 | `mdp/events.py:558` 부근 (`_peg_leg_default_joint_pos_ref`) | peg-leg 리셋이 `default_joint_pos` 를 lock 각으로 덮어쓰기 **전에** 최초 1회 원본을 저장. 관측 함수는 이 스냅샷을 기준으로 뺀다 |
| 플래그 기본값 | `scripts/rsl_rl/train_phase2.sh:130` | `export GO1_ABS_JOINT_OBS="${GO1_ABS_JOINT_OBS:-1}"` — 모든 phase 런처가 train_phase2.sh 를 경유하므로 파이프라인 전체에서 기본 활성 |

### 왜 추가했나

부상(부목)은 calf 관절을 lock 각도에 고정하는 방식으로 구현되는데, 리셋 이벤트가
`default_joint_pos` 자체를 lock 각으로 재작성한다. 표준 관측 `joint_pos_rel` 은
`joint_pos − default_joint_pos` 이므로 부상 calf 채널이 **에피소드 내내 ≈0** 이 되어,
부목 길이가 proprioception 에서 관측 불가능해진다.

- **실측 근거**: student 의 splint-length 추정 probe **R² = 0.00** (평균 예측과 동일 수준).
  Phase 3 distillation 의 전제(teacher latent 를 proprioception history 로 추정)가 무너짐.
- **물리적 정당성**: 실기체 관절 엔코더는 절대각을 읽을 뿐 "부상 후 새 default" 라는
  개념이 없다. 이 소거는 물리가 아니라 시뮬레이션 인공물이다.
- **효과**: 부상 전 nominal(calf −1.5 rad)을 빼므로 `(lock각 − nominal)` 이 상수로 노출된다.
  부목 길이와 1:1 대응 (`events.py` 의 코사인 법칙 변환, thigh = calf = 0.213 m):
  - 부목 0.20 m (severe) → lock −2.16 rad → 관측값 **−0.66**
  - 부목 0.30 m (mild) → lock −1.58 rad → 관측값 **−0.08**
  - 정상 다리 → nominal ≈ default 이므로 기존 관측과 사실상 동일 (보행 스윙)

기존 `joint_pos_rel` 은 제거하지 않고 유지한다. 부상 calf 에서 두 항은 상보적이다:
`joint_pos_rel` = q − lock = **처짐(하중 프록시, 동적)**, `calf_pos_nominal_rel` =
q − nominal = **부목 심각도(정적)**. 정적/동적 신호가 분리된 형태로 들어간다.

### policy 그룹 52차원 레이아웃 (concat 순서)

| 인덱스 | 항목 | 차원 |
|---|---|---|
| 0–2 | base_lin_vel | 3 |
| 3–5 | base_ang_vel | 3 |
| 6–8 | projected_gravity | 3 |
| 9–11 | generated_commands (vx, vy, ωz) | 3 |
| 12–23 | joint_pos_rel | 12 |
| 24–35 | joint_vel_rel | 12 |
| 36–47 | last_action | 12 |
| **48–51** | **calf_pos_nominal_rel (신규)** | **4** |

관절 12개 순서는 Isaac Lab per-TYPE 순서:
`[FL_hip, FR_hip, RL_hip, RR_hip, FL_thigh, …, RR_thigh, FL_calf, …, RR_calf]`
(per-leg `leg*3+2` 공식 아님 — 주의).

---

## 변경 2 — privileged_obs 3 → 7: 부상 위치 one-hot 인코딩

### 어디에 추가되었나

| 코드 | 위치 | 내용 |
|---|---|---|
| 관측 함수 | `mdp/observations.py:44` (`peg_leg_one_hot`) | `[FL, FR, RL, RR, injured_flag]` 5차원 one-hot |
| 그룹 정의 | `go1_lab_env_cfg.py:53-69` (`Go1LabPrivilegedObsCfg`) | 기본: index(1) + splint_length(1) + foot_friction(1) = 3차원 |
| one-hot 치환 | `go1_lab_env_cfg.py:231-232` | `GO1_INJURY_ONEHOT=1` 이면 scalar `peg_leg_index` 를 `peg_leg_one_hot` 으로 교체 → 5+1+1 = 7차원 |
| 플래그 설정 | `baselines/launch_phase1.sh:19`, `launch_warmstart_compar.sh:42` 등 | 모든 런처가 `GO1_INJURY_ONEHOT=1` 명시 |

### 왜 추가했나

scalar index(0=정상, 1~4=부상 다리)는 연속값 하나로 5개 이산 조건을 인코딩하므로
conditioning 신호가 약하다. **실측**: teacher 가 다리별로 구분된 antalgic 보행을 학습하지
못하고 지배 모드 하나로 붕괴 (FL 만 하중, FR/RL/RR 부상 조건 포기). one-hot 은 다리별
조건화를 **선형**으로 만들어 (RMA 계열에서 이산 privileged 인자의 표준 인코딩) 부상
위치별로 구분된 반응을 학습할 수 있게 한다.

### privileged 7차원 레이아웃 (전체 concat 기준 인덱스 52–58)

| 인덱스 | 항목 | 차원 | 정상일 때 값 |
|---|---|---|---|
| 52–56 | peg_leg_one_hot [FL, FR, RL, RR, injured_flag] | 5 | 0 |
| 57 | peg_leg_splint_length (m) | 1 | 0 |
| 58 | peg_leg_foot_friction | 1 | 0 |

Phase 1(healthy)에서도 이 그룹을 항상 등록해 값 0 으로 유지한다
(`go1_lab_env_cfg.py:222`) — phase 1 → 2 warm-start 차원 호환을 위해서다.

---

## 네트워크 연결 (누가 몇 차원을 받나)

`agents/rsl_rl_ppo_cfg.py`:

```python
# Teacher (Phase 1/2), rsl_rl_ppo_cfg.py:45
obs_groups = {"policy": ["policy", "privileged_obs"],   # actor  = 52+7 = 59
              "critic": ["policy", "privileged_obs"]}   # critic = 59

# Distillation (Phase 3), rsl_rl_ppo_cfg.py:95
obs_groups = {"student": ["policy"],                    # student = 52 (LSTM)
              "teacher": ["policy", "privileged_obs"]}  # teacher = 59 (동결)
```

MLP 구조: 59 → 512 → 256 → 128 → 12 (ELU). action 12차원은 관절 위치 목표:
`target = default_joint_pos + 0.25 × action`.

---

## 검증 방법

```bash
# 1. 체크포인트 입력 차원 (59 여야 함)
python3 -c "import torch; d=torch.load('models/phase1_mlp_s42/model_5999.pt',
map_location='cpu', weights_only=False); print(d['model_state_dict']['actor.0.weight'].shape)"
# → torch.Size([512, 59])

# 2. 학습 당시 env 설정에 calf_pos_abs 존재 확인
grep -n "calf_pos_abs" models/phase1_mlp_s42/params/env.yaml

# 3. export 산출물의 그룹별 레이아웃
# models/phase1_mlp_s42/exported/IO_descriptors.yaml (policy 52 + privileged_obs 7)
```

`IO_descriptors.yaml` 은 `scripts/rsl_rl/export_policy_onnx.py` 의
`_export_io_descriptors()` 가 생성하며, 배포 측이 59차원 레이아웃을 재구성하는 기준
문서다 (policy 그룹만 내보내는 기본 동작을 privileged_obs 포함으로 확장했음).

---

## 주의사항

1. **차원 플래그는 전 phase 에서 일치해야 한다.** `GO1_ABS_JOINT_OBS` 와
   `GO1_INJURY_ONEHOT` 이 phase 1/2/3/eval 중 하나라도 다르면 체크포인트 로드가
   실패하거나 (차원 불일치) 관측 의미가 어긋난다. 현재는 train_phase2.sh 기본값과
   런처 명시 설정으로 전부 일치한다.
2. **관측 구성을 바꾸면 전 phase 재학습이다.** calf 항 제거, one-hot 해제,
   `default_joint_pos` 재작성 삭제(→ joint_pos_rel 분포 변화) 모두 해당.
3. **실기체 배포 시** privileged 7차원은 센서가 아니라 실험 조건 상수로 주입한다
   (장착한 부목의 위치 one-hot / 실측 길이 / 마찰 추정치). 부목 길이 입력이 실제와
   다르면 conditioning 이 어긋나 보행이 거칠어진다.
4. **배포 시 joint_pos_rel 의 부상 calf 채널은 `q − lock각`으로 계산해야 한다**
   (학습 분포가 그렇다). 즉 현재 관측 구성은 배포 코드가 lock 각을 알아야 한다 —
   물리 부목 실험에서는 실측 가능하므로 성립한다.
