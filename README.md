# go1_compar — antalgic Go1 파이프라인 + 환부 보상항 대조군 (self-contained)

**서버에서 clone → 실행**할 수 있도록 환경·학습·평가 코드를 모두 포함한 self-contained repo.

**제안 알고리즘과 비교군은 별개의 워크플로우가 아니라 하나의 파이프라인이다.**
Phase1(정상보행) → Phase2(warm-start + 부상 + 통각)라는 동일한 커리큘럼(§4.6)을 돌리되,
**환부 다리 보상항 하나만** 바꿔 세 갈래로 나뉜다 (§2.3):

- **antalgic** — 통각 penalty. **이것이 제안 알고리즘**이다.
- **fault-tolerant** — 통각 없음. 비교군.
- **symmetry** — 좌우대칭 penalty. 비교군.

셋 다 같은 phase1 체크포인트에서 출발하고 환경·구조·PPO·커리큘럼이 전부 동일하므로,
결과 차이는 환부 보상항에만 귀속된다.

## 파이프라인: Phase 1 → Phase 2 warm-start (제안 알고리즘)

**설계**: 부상 보행은 정상 보행 위에 얹히는 것이라는 관찰을 그대로 학습 절차로 옮긴다.
먼저 정상 보행(phase1)을 학습해 생리적 gait를 확보하고, 그 정책에서 warm-start해
부상(기능적 부목)과 통각(eq.4)을 얹는다. 걷는 습관이 유지된 채 통각이 하중만 줄이므로
**부분하중 antalgic gait**가 나온다. 보조 장치는 두 가지: `GO1_PHASE2_GAIT_TUNING=1`
(대칭 강요 없는 anti-buzz 정규화)로 고주파 버즈를 막고, 중간 세기의 viability floor로
부상다리를 gait에 붙잡아 완전 비사용(non-use)으로 도망가지 못하게 한다.

```bash
cd baselines
# 1) (선택) 정상 보행 phase1 직접 학습 — 부상·통증 없음. ~2.6 Hz walk.
#    번들 체크포인트(models/)가 있으면 2)가 이걸 쓰지 않는다. 아래 주의 참조.
./launch_phase1.sh 42                       # → logs/.../phase1_mlp_s42/model_5999.pt

# 2) phase1에서 warm-start → 부상(기능적 부목)+통각(eq.4) 얹어 antalgic 학습.
#    제안 알고리즘 = 이 antalgic 팔. faulttol/symmetry 는 동일 파이프라인의 비교군.
./launch_warmstart_compar.sh antalgic 42    # 기본: 번들 models/phase1_mlp_s42
#   직접 학습한 phase1을 쓰려면 경로를 3번째 인자로 명시한다:
./launch_warmstart_compar.sh antalgic 42 ../scripts/rsl_rl/logs/rsl_rl/unitree_go1_rough_teacher/<run>/model_5999.pt

# 3) 평가 (gait 주파수·부상다리 duty·GRF)
PHASE2_RUN_NAME=phase2_ws_antalgic_s42 \
GO1_BIOMECH_DUMP=$PWD/../scripts/rsl_rl/biomech/ws.npz \
  ../scripts/rsl_rl/analyze_phase2_balanced.sh   # (AGENT=rsl_rl_teacher_mlp_cfg_entry_point)
```
**판정 기준**: normal gait 2–3 Hz, 부상다리 duty 0.3–0.5(부분하중), 4다리 안정.

> **phase1은 seed와 무관하게 고정이다 — 의도된 통제.**
> `launch_warmstart_compar.sh`는 번들 `models/phase1_mlp_s42/model_5999.pt` 를 항상 warm-start
> 소스로 쓴다(경로를 인자로 넘기면 그게 우선). 모든 패러다임·모든 seed가 **동일한 초기
> 정책**에서 출발하므로, 패러다임 간 차이를 오직 환부 보상항에 귀속시킬 수 있는
> matched design이 된다 — phase1이 우연히 좋았는지 여부가 비교에 개입하지 못한다.
>
> seed가 바꾸는 것은 **환경 랜덤화**(domain rand, 부상 다리 배정, 부목 길이, 초기 상태)와
> **PPO 탐색·미니배치 셔플**이다(`--seed` → `env_cfg.seed` + `agent_cfg.seed`). 따라서
> n-seed는 서로 다른 phase2 학습 n회가 맞다. 다만 논문에서는 편차의 범위를 정확히 쓸 것:
> "공통 phase1에서 출발한 phase2 seed n회"이지 파이프라인 전체 반복이 아니다.
>
> phase1 자체를 다시 만들 때만 `launch_phase1.sh` 를 쓰고 그 결과로 `models/` 를 교체한다.
> (`./launch_phase1.sh 43` 처럼 다른 seed로 돌려도 번들이 있는 한 아무것도 그걸 읽지 않는다.)

---

## 비교군: 같은 파이프라인, 환부 보상항만 교체

제안 방법(antalgic)이 부상동물 biomechanics를 재현하는 것이 **통각 보상 때문**임을,
**동일한 환경·구조에서 환부 다리 보상항만 바꾼** 3개 패러다임 비교로 입증한다.
위 파이프라인의 2단계에서 인자만 `faulttol`/`symmetry` 로 바꾸면 비교군이 된다.

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
  train_phase2.sh          phase1·phase2 학습의 단일 진입점 (이름과 달리 phase1도 이걸로
                           학습한다 — phase1 = 부상확률 0 · 통각 0인 특수 케이스).
                           학습 로직은 없고, train.py 가 읽을 GO1_* 환경변수를
                           `${VAR:-default}` 로 세팅할 뿐인 "bash로 쓴 설정 파일".
                           launcher가 환경변수로 덮어쓴다. (직접 실행하지 않음)
                           소스 기본값과 같은 값은 중복 기재하지 않는다. 단 부상모델·
                           커리큘럼·eq.3 가중치처럼 실험 설계를 드러내는 값은 남긴다.
  train.py                 Isaac Lab / rsl-rl 학습 스크립트
  analyze_phase2_balanced.sh, analyze_student.py, biomech_analyze.py,
  extract_paper_metrics.py, aggregate_nseed.py
models/phase1_mlp_s42/   번들된 clean phase1 체크포인트 (2.6Hz walk, warm-start 소스).
                         서버에서 재학습 없이 바로 warm-start 가능 (git 추적됨).
baselines/               실행 스크립트 (모두 portable: systemd 없이 foreground)
  launch_phase1.sh       [1단계]  정상보행 phase1 학습 (teacher-MLP). 번들 models/ 를
                                  다시 만들 때만 쓴다 — 평소엔 실행할 필요 없음.
  launch_warmstart_compar.sh      [2단계] 팔 1개를 phase1에서 warm-start해 학습.
                                  인자에 따라 제안/비교군이 갈린다:
                                    antalgic  → [제안]   통각 penalty (제안 알고리즘 본체)
                                    faulttol  → [비교군] 통각 없음
                                    symmetry  → [비교군] 좌우대칭 penalty
  run_baselines.sh       [공통] 제안+비교군 3개 학습 → 자동 평가·비교 (detached)
  run_nseed_compar.sh    [공통] 위를 n-seed로 반복 → 패러다임별 mean±SD·95% CI
  eval_compar.sh         [공통] 3개 저속평가 → biomech 덤프 → 비교표
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
# 제안(antalgic) + 비교군 2개 학습 + 자동 비교 (seed 42). detached 권장:
nohup ./run_baselines.sh 42 > run42.log 2>&1 &
#   → 번들 phase1(models/)에서 3개 모두 warm-start → phase2_ws_<paradigm>_s42
#   → 완료 시 baselines/compare_result.txt 에 비교표

# 개별 실행:
./launch_warmstart_compar.sh antalgic 42   # 제안 알고리즘 (foreground; nohup/tmux 권장)
./launch_warmstart_compar.sh faulttol 42   # 비교군
./launch_warmstart_compar.sh symmetry 42   # 비교군
./eval_compar.sh 42                        # 학습 완료 후 평가+비교

# n-seed 통계 (패러다임별 mean±SD, 95% CI):
nohup ./run_nseed_compar.sh "42 43 44 45 46" > nseed.log 2>&1 &
```

옵션: `PARALLEL=1 ./run_baselines.sh 42` (작은 GPU에서 순차 실행),
`NUM_ENVS=1024 ...`, `PHASE2_MAX_ITER=12000 ...`.

## 설계 노트 (검토 요망)

- **viability floor를 3개 모두에 포함** → 모든 패러다임이 환부를 "사용"하므로
  use-vs-nonuse가 아니라 **보행 패턴**을 공정 비교. 논문에 이 결정 명시.
- **모든 seed가 phase1을 공유하는 것은 통제 장치** — 세 패러다임이 동일한 초기 정책에서
  출발하므로 차이가 환부 보상항에 귀속된다(위 §Phase1→Phase2 주석 참조). 논문에는
  편차의 범위만 정확히 적으면 된다: "공통 phase1에서 출발한 phase2 seed n회".
- **symmetry 가중치 −2.0은 시작값** — 대칭이 실제로 강제되도록 −1~−5 조정 확인 권장.
- 학습/평가 산출물은 `scripts/rsl_rl/logs`·`scripts/rsl_rl/biomech` 에 저장(gitignore).

지표·통계는 `baselines/compare_3paradigm.py` 출력 참조.
