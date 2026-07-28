# go1_compar — antalgic Go1 pipeline + 3-paradigm baseline (self-contained)

**서버에서 clone → 실행**할 수 있도록 환경·학습·평가 코드를 모두 포함한 self-contained repo.
두 가지 워크플로우:
1. **Phase1 → Phase2 warm-start 파이프라인** (아래) — 정상 보행을 먼저 학습하고 그 위에
   부상+통각을 얹어 **생리적 gait를 유지한 antalgic 보행**을 학습. (논문 §4.6 커리큘럼)
2. **3-paradigm baseline 비교** — 환부 보상항만 바꾼 antalgic/fault-tol/symmetry 비교 (§2.3).

## Phase 1 → Phase 2 warm-start 파이프라인 (gait-faithful)

**배경**: from-scratch로 phase2를 학습하면 고주파(12–17 Hz) **바운딩**이나 부상다리
**비사용(non-use)** 같은 비생리적 gait로 붕괴한다. 정상 보행(phase1)을 먼저 학습하고
warm-start로 이어받으면, 걷는 습관이 유지된 채 통각이 하중만 줄여 **부분하중 antalgic
gait**가 나온다. 핵심은 `GO1_PHASE2_GAIT_TUNING=1`(대칭 강요 없는 anti-buzz 정규화)로
버즈를 막고, 중간 세기의 viability floor로 부상다리를 gait에 붙잡는 것.

```bash
cd baselines
# 1) 정상 보행 phase1 (teacher-MLP, 부상·통증 없음) — warm-start 호환. ~2.6 Hz walk.
./launch_phase1.sh 42                       # → logs/.../phase1_mlp_s42/model_5999.pt

# 2) phase1에서 warm-start → 부상(기능적 부목)+통각(eq.4) 얹어 antalgic 학습
./launch_warmstart_phase2.sh 42             # phase1 체크포인트 자동탐색
#   또는 명시: ./launch_warmstart_phase2.sh 42 /path/to/phase1_mlp_s42/model_5999.pt

# 3) 평가 (gait 주파수·부상다리 duty·GRF)
GO1_BIOMECH_DUMP=$PWD/../scripts/rsl_rl/biomech/ws.npz \
  ../scripts/rsl_rl/analyze_phase2_balanced.sh   # (AGENT=rsl_rl_teacher_mlp_cfg_entry_point)
```
**판정 기준**: normal gait 2–3 Hz, 부상다리 duty 0.3–0.5(부분하중), 4다리 안정.

---

## 3-paradigm baseline 비교

제안 방법(antalgic)이 부상동물 biomechanics를 재현하는 것이 **통각 보상 때문**임을,
**동일한 환경·구조에서 환부 다리 보상항만 바꾼** 3개 패러다임 비교로 입증한다.

## 3개 패러다임 (환부 다리 보상항만 다름)

| 패러다임 | 환부 보상 | 설정 | 기대 direction-of-change |
|---|---|---|---|
| **antalgic** (제안) | 통각 penalty `C_pain(F)` | `GO1_PAIN_WEIGHT=-0.05` | **>80%** |
| **fault-tolerant** | 없음 (alive bonus만) | `GO1_PAIN_WEIGHT=0` | **<40%** |
| **symmetry** | 좌우대칭 penalty `-λ‖q−M(q)‖²` | `GO1_SYMMETRY_PENALTY_WEIGHT=-2.0` | **<50%** |

**공통(동일)**: 환경, 부상모델(기능적 부목), RMA 구조, PPO, PD(Kp20/Kd0.5), domain
randomization, 커리큘럼, load-bearing viability floor, 속도추종·에너지 보상.
유일 변수 = 환부 보상항.

## 구조

```
source/go1_lab/          Isaac Lab 확장 (환경 + 알고리즘 + Go1/pegleg USD)
scripts/rsl_rl/          학습·평가 파이프라인
  train_phase2.sh          모든 학습의 단일 진입점. 모든 보상/부상/커리큘럼 기본값이
                           `${VAR:-default}` 한 곳에 평평하게 모여 있고, baselines/의
                           launcher가 환경변수로 덮어쓴다. (직접 실행하지 않음)
  train.py                 Isaac Lab / rsl-rl 학습 스크립트
  analyze_phase2_balanced.sh, analyze_student.py, biomech_analyze.py,
  extract_paper_metrics.py, aggregate_nseed.py
models/phase1_mlp_s42/   번들된 clean phase1 체크포인트 (2.6Hz walk, warm-start 소스).
                         서버에서 재학습 없이 바로 warm-start 가능 (git 추적됨).
baselines/               실행 스크립트 (모두 portable: systemd 없이 foreground)
  launch_phase1.sh          [파이프라인] 정상보행 phase1 (teacher-MLP, warm-start 호환)
  launch_warmstart_phase2.sh[파이프라인] phase1→phase2 warm-start + 부상 + 통각
  launch_warmstart_compar.sh[baseline★] 3패러다임을 **같은 phase1에서 warm-start** (권장)
  launch_compar.sh       [baseline] 한 패러다임 teacher 1개 학습 (from-scratch, 구버전)
  run_baselines.sh       [baseline] 3개 패러다임 학습 → 자동 평가·비교 (detached).
                         기본 launcher = launch_warmstart_compar.sh
  run_nseed_compar.sh    [baseline] 위를 n-seed로 반복 → 패러다임별 mean±SD·95% CI
  eval_compar.sh         [baseline] 3 패러다임 저속평가 → biomech 덤프 → 비교표
  compare_3paradigm.py   direction/GRF/SI 비교표
```

## 설정 (서버)

1. **Isaac Lab 5.1** 설치 (NVIDIA 문서 참조) + 해당 python 환경 활성화.
2. 이 확장 설치:
   ```bash
   python -m pip install -e source/go1_lab
   ```
   태스크는 `Template-Go1-Lab-v0` 로 등록됨.
3. `python3 train.py` 가 Isaac Lab python을 가리키도록 **환경을 먼저 활성화**한 뒤 실행.

## 실행

```bash
cd baselines
# 3개 패러다임 학습 + 자동 비교 (seed 42). detached 권장:
nohup ./run_baselines.sh 42 > run42.log 2>&1 &
#   → 번들 phase1(models/)에서 3개 모두 warm-start → phase2_ws_<paradigm>_s42
#   → 완료 시 baselines/compare_result.txt 에 비교표

# 개별 실행:
./launch_warmstart_compar.sh antalgic 42   # (foreground; nohup/tmux로 백그라운드)
./launch_warmstart_compar.sh faulttol 42
./launch_warmstart_compar.sh symmetry 42
./eval_compar.sh 42                        # 학습 완료 후 평가+비교

# n-seed 통계 (패러다임별 mean±SD, 95% CI):
nohup ./run_nseed_compar.sh "42 43 44 45 46" > nseed.log 2>&1 &
```

옵션: `PARALLEL=1 ./run_baselines.sh 42` (작은 GPU에서 순차 실행),
`NUM_ENVS=1024 ...`, `PHASE2_MAX_ITER=12000 ...`.

from-scratch(구버전) 비교로 되돌리려면:
`LAUNCHER=launch_compar.sh RUN_PREFIX=phase2_cmp ./run_baselines.sh 42`

## 설계 노트 (검토 요망)

- **viability floor를 3개 모두에 포함** → 모든 패러다임이 환부를 "사용"하므로
  use-vs-nonuse가 아니라 **보행 패턴**을 공정 비교. 논문에 이 결정 명시.
- **symmetry 가중치 −2.0은 시작값** — 대칭이 실제로 강제되도록 −1~−5 조정 확인 권장.
- 학습/평가 산출물은 `scripts/rsl_rl/logs`·`scripts/rsl_rl/biomech` 에 저장(gitignore).

지표·통계는 `baselines/compare_3paradigm.py` 출력 참조.
