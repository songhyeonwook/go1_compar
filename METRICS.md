# 비교군에 필요한 지표 정리 (논문 §2.3 / §2.5)

3-paradigm 비교의 목적 = **"antalgic만 부상동물 패턴을 재현한다"** 를 입증.
아래 지표를 3개 패러다임(antalgic / fault-tolerant / symmetry)에서 각각 측정해
비교한다. 모두 `eval_compar.sh` → `compare_3paradigm.py`가 자동 산출.

---

## 1. ★ 주 지표 (primary) — direction-of-change agreement %

부상동물 문헌 기대부호(GRF↓, stance 연장, contra/ipsi impulse↑, CoM intact측)와의
**부호 일치율**. 패러다임 간 **분리**가 핵심.

| 패러다임 | 기대값 | 판정 |
|---|---|---|
| antalgic | **> 80%** | 부상동물 패턴 재현 |
| fault-tolerant | **< 40%** | 재현 실패 |
| symmetry | **< 50%** | 재현 실패 |

→ 이 분리가 나오면 "통각 보상이 antalgic 재현의 원인"이 baseline 대비 입증됨.
(antalgic 참조값은 n=10에서 82.5 ± 5.8%.)

---

## 2. 보조 지표 (secondary) — 패러다임별 정량값

| 지표 | 의미 | 기대 대조 |
|---|---|---|
| **injured-limb GRF 감소 %** | 환부 off-loading 정도 | antalgic 높음(~82%), fault-tol 낮음(적재), symmetry 중간 |
| **SI (eq.7) %** | healthy-vs-affected 비대칭 | antalgic 높음(~140%), 나머지 낮음 |
| **injured-limb stance/duty** | 접촉시간 | antalgic 연장(weight-bearing), symmetry 강제대칭 |
| **contra/ipsi vertical impulse %** | 보상 재분배 | antalgic만 동물 패턴 방향 |
| **CoM lateral shift (cm)** | 무게중심 이동 | antalgic만 intact측 |
| **survival %** | 이동 성공률 | 3개 모두 locomote (분리는 패턴에서) |

---

## 3. 통계 (n-seed, §4.11)

각 패러다임을 **n≥5 (권장 10) 독립 시드**로 학습 → 지표별 **mean ± SD + 95% CI**.
패러다임 간 차이 검정:

- **Mann–Whitney U** (antalgic vs 각 baseline), **Holm–Bonferroni** 보정.
- **Cliff's δ** (효과크기).
- primary(direction-of-change)에서 antalgic > baseline 이 유의(p<0.05)해야 함.

> 파일럿(n=1)으로 먼저 **분리 방향**을 확인한 뒤 n-seed로 확장하는 것을 권장
> (분리가 n=1에서 안 나오면 symmetry 가중치 등 setup부터 조정).

---

## 4. 논문 표 형식 (§2.3 또는 §2.5)

`compare_3paradigm.py`가 출력하는 표를 그대로 사용:

```
paradigm          GRFred%  SI(eq7)%  direction%  expect
antalgic             82.x     140.x        82.x   >80
faulttol             xx.x      xx.x        xx.x   <40
symmetry             xx.x      xx.x        xx.x   <50
```

여기에 각 셀을 **mean ± SD (n=N)** 로 채우고, direction 열에 **Mann–Whitney p값**을
각주로 추가하면 논문 §2.3 비교표 완성.

---

## 5. 산출 방법 요약

| 지표 | 산출 |
|---|---|
| direction-of-change, GRF, SI | `compare_3paradigm.py` (biomech npz) |
| stance/duty, impulse, CoM | `go1_peg/scripts/rsl_rl/extract_paper_metrics.py` (동일 npz) |
| mean±SD, CI | 시드별 반복 후 `aggregate_nseed.py` 방식 |
| Mann–Whitney U, Cliff's δ | scipy.stats.mannwhitneyu + 별도 계산 (n-seed 후) |

**핵심 메시지**: 세 패러다임이 모두 걷지만(survival 유사), **direction-of-change에서만
antalgic이 >80%로 분리** → 통각 보상이 부상동물 biomechanics 재현의 **원인**임을
baseline 대비로 입증.
