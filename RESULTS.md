# 비교군 구현 내용 + 결과 (3-paradigm)

세 패러다임은 **동일한 환경·부상모델·RMA 구조·PPO·PD·DR·커리큘럼·viability floor**
를 쓰고, **환부 다리 보상항 2개 env-var만** 다르다. (`baselines/launch_compar.sh`)

## 1. 패러다임별 실제 설정 (환부 보상항만 다름)

| 패러다임 | `GO1_PAIN_WEIGHT` | `GO1_SYMMETRY_PENALTY_WEIGHT` | 의미 |
|---|---|---|---|
| **antalgic** (제안) | **−0.05** | 0.0 | 통각 penalty `C_pain(F)` (eq.4) |
| **fault-tolerant** | **0.0** | 0.0 | 통각 제거, alive bonus만 |
| **symmetry** | 0.0 | **−2.0** | 좌우대칭 penalty `−λ‖q−M(q)‖²` |

공통(3개 동일): `GO1_NO_WARMSTART=1 GO1_INJURY_ONEHOT=1 GO1_PD_ACTUATOR=1
GO1_PD_KP=20 GO1_PD_KD=0.5 GO1_STRICT_TERMINATIONS=1 GO1_BAD_ORIENTATION_LIMIT=0.8
GO1_DOMAIN_RAND=1 GO1_CMD_VX_MIN/MAX=0.10/0.30 GO1_FLAT_ORIENTATION_WEIGHT=-2.0
GO1_SURVIVAL_BONUS_WEIGHT=1.0 GO1_SPLINT_CALF_ANGLE=-1.5 GO1_SPLINT_CALF_STIFFNESS=12
GO1_PEG_HIP_TORQUE_SCALE=1.0 GO1_INJURED_FORCE_NONUSE_WEIGHT=-0.5 (min 4N)
GO1_INJURED_DUTY_NONUSE_WEIGHT=-1.0 (min duty 0.30)` — 각 18k iter, 2048 env.

구현 위치:
- 통각 보상 `penalty_pain` (eq.4): `source/go1_lab/.../mdp/rewards.py`
- 대칭 penalty `penalize_joint_mirror_asymmetry` (§4.7): `.../mdp/rewards.py`, env-var
  `GO1_SYMMETRY_PENALTY_WEIGHT` (`.../go1_lab_env_cfg.py`)
- alive bonus `survival_bonus`: `.../go1_lab_env_cfg.py`

## 2. 결과 (seed 42, n=1 pilot)

teacher 저속 직진(vx=0.3) 평가:

| paradigm | GRF감소% | SI(eq7)% | direction% | injured 행동 |
|---|---|---|---|---|
| **antalgic** | **80** | **133** | **87.5** | 4다리 모두 off-load (75–85%) |
| **fault-tolerant** | **−31** | **8** | 58.3 | off-load 안 함; FL 부상 시 **199.9N 과적재**(정상 60N) |
| **symmetry** | 14 | 18 | 70.8 | 앞다리 off-load 안 함(강제대칭 −7/−12%) |

**핵심**: **antalgic만 환부를 protective off-load(GRF −80%) + 높은 antalgic 비대칭
(SI 133%)**. fault-tolerant는 오히려 과적재, symmetry는 대칭 강제 → 둘 다 antalgic 미재현.
→ **통각 보상이 antalgic biomechanics 재현의 원인**을 baseline 대비로 입증.

**주 discriminator = GRF감소 + SI** (극적 분리). direction-of-change는 6-부호 일치라
coarse(baseline 58–71%) → 순서(ordering)만 보조로.

## 3. 재현 (서버)

```bash
cd baselines
nohup ./run_baselines.sh 42 > run42.log 2>&1 &   # 3패러다임 학습 + 비교 → compare_result.txt

# n-seed 통계(패러다임별 mean±SD)로 확장:
nohup ./run_nseed_compar.sh "42 43 44 45 46" > nseed.log 2>&1 &
```

지표 정의·통계는 `METRICS.md` 참조.
