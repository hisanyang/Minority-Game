# Minority Game

이 저장소는 binary-choice minority game과 mixed-memory/cognitive-capacity minority game을 시뮬레이션하고, 각 실험 결과를 그림으로 저장하는 코드입니다. 중심 스크립트는 `260415_Minority_Dynamic.py`입니다.

## Main script

`260415_Minority_Dynamic.py`는 다음 흐름으로 작동합니다.

1. 에이전트별 전략을 무작위로 생성합니다.
2. 각 에이전트가 현재 history에 대해 가장 높은 score를 가진 전략을 선택합니다.
3. 전체 행동 합계 `A`를 계산합니다.
4. minority rule에 따라 winning action을 정합니다.
5. 선택된 전략들의 score를 업데이트합니다.
6. equilibrium 이후 구간에서 volatility, success rate, group imbalance 등을 기록합니다.
7. 결과 그림을 `Figures/` 폴더에 저장합니다.

그림 저장은 `save_figure()` 함수가 담당합니다. 이 함수는 `Figures/` 폴더가 없으면 자동으로 만들고, 모든 새 그림을 그 폴더 안에 저장합니다.

## Key parameters

- `N`: 전체 에이전트 수입니다.
- `M`: 단일 집단 실험에서 history 길이 또는 cognitive capacity입니다.
- `M1`, `M2`: mixed-M 실험에서 두 집단의 cognitive capacity입니다.
- `S`: 에이전트별 전략 개수입니다. 현재 기본값은 `2`입니다.
- `TOTAL_STEPS`: 한 trial에서 실행하는 전체 time step 수입니다.
- `EQ_STEPS`: 통계 계산에서 제외할 초기 burn-in step 수입니다.
- `TRIALS`: 각 `N` 값마다 반복하는 Monte Carlo trial 수입니다.
- `MASTER_SEED`: 재현 가능한 난수 생성을 위한 seed입니다.
- `N_values`: 실험에서 훑는 전체 에이전트 수 목록입니다.
- `EXPERIMENT_MODE`: 어떤 실험을 실행할지 고르는 스위치입니다.

## Core functions

### `run_binary_choice_game`

기본 binary-choice game을 실행합니다. 모든 에이전트가 같은 `M`과 `S`를 갖고, 현재 history에 대해 가장 높은 score의 전략을 사용합니다.

반환값은 volatility와 average success rate입니다. `return_records=True`이면 전체 time series도 함께 반환합니다.

### `run_mixed_m_game`

두 집단이 서로 다른 cognitive capacity를 갖는 mixed-M minority game을 실행합니다. Group 1은 `M1`, Group 2는 `M2`를 사용하며, 두 집단 모두 adaptive strategy learning을 합니다.

주요 결과는 각 집단의 average success rate입니다.

### `run_mixed_m_game_large_m_balanced`

Group 1은 adaptive하게 행동하지만, Group 2는 학습하지 않고 매 시점 균형 잡힌 행동을 하도록 만든 비교 실험입니다. Group 2 안에서 `+1`과 `-1` 행동 수를 최대한 비슷하게 맞춥니다.

### `run_mixed_m_game_large_m_random`

Group 1은 adaptive하게 행동하지만, Group 2는 학습하지 않고 매 시점 무작위 행동을 합니다.

### `run_mixed_m_game_diagnostics`

mechanism figure를 만들기 위한 진단용 함수입니다. 각 집단의 history state 방문 횟수, normalized imbalance, random-choice benchmark 대비 excess imbalance를 계산합니다.

## Experiment modes

### Mode 1: Baseline Minority Game

단일 cognitive capacity `M_FIXED`를 가진 기본 minority game입니다.

이 모드는 두 종류의 그림을 만들 수 있습니다.

- Baseline volatility vs. `2^M / N`
- Baseline average success rate vs. `2^M / N`
- Typical run time series with zoom

현재 코드에서는 `RUN_MAIN_MODE1_FIGURES`와 `RUN_TYPICAL_RUN_FIGURE` 플래그로 어떤 그림을 만들지 고릅니다.

### Mode 2: Mixed-M Minority Game, Both Adaptive

서로 다른 cognitive capacity를 가진 두 집단을 비교합니다. Group 1은 `M1`, Group 2는 `M2`를 사용하고, 두 집단 모두 adaptive strategy learning을 합니다.

이 실험의 핵심 질문은 낮은 capacity 집단과 높은 capacity 집단 중 어느 쪽이 더 높은 average success rate를 얻는가입니다.

출력 그림은 `Figure2_Average_Success_Rate_by_Cognitive_Capacity (...).png` 형식으로 저장됩니다.

### Mode 3: M1 Adaptive vs. M2 Balanced Non-Adaptive

Group 1은 낮은 `M1`을 가진 adaptive agents입니다. Group 2는 높은 `M2` 집단으로 설정되지만, 실제로는 학습하지 않고 매 시점 `+1`과 `-1`을 거의 같은 수로 내는 balanced non-adaptive benchmark로 행동합니다.

종속변수는 각 집단 안의 `Minority - Majority` 값입니다. 코드에서는 `-abs(A_g)`로 기록됩니다.

### Mode 4: M1 Adaptive vs. M2 Random Non-Adaptive

Group 1은 adaptive agents입니다. Group 2는 non-adaptive random agents로, 매 시점 무작위로 `+1` 또는 `-1`을 선택합니다.

Mode 3과 마찬가지로 각 집단의 `Minority - Majority` 값을 비교합니다. balanced benchmark가 아니라 random benchmark를 쓰는 점이 차이입니다.

### Mode 5: Mechanism Figure Panels

mixed-M 결과가 왜 나타나는지 설명하기 위한 mechanism figure를 만듭니다.

Panel A는 low-M과 high-M 집단의 history state 반복 방문 정도를 비교합니다. 낮은 `M`은 가능한 history 수가 작기 때문에 같은 state를 더 자주 반복 방문합니다.

Panel B는 각 집단의 normalized within-group imbalance `|A_g| / N_g`를 random-choice benchmark와 비교합니다.

Panel C는 random-choice benchmark 대비 excess imbalance를 계산합니다. 이 값이 0보다 크면 해당 집단의 행동이 무작위보다 더 군집되어 있다는 뜻입니다.

## Outputs

모든 새 그림은 `Figures/` 폴더에 저장됩니다. 기존 코드에서 직접 `fig.savefig(...)`를 호출하던 부분은 `save_figure(...)`를 사용하도록 정리되어 있습니다.

대표 산출물은 다음과 같습니다.

- `Figures/Figure2_Average_Success_Rate_by_Cognitive_Capacity (...).png`
- `Figures/Figure3_PanelA_History_Repetition.png`
- `Figures/Figure3_PanelB_Imbalance_With_Random_Benchmark.png`
- `Figures/Figure3_PanelC_Excess_Imbalance.png`

## How to run

`260415_Minority_Dynamic.py` 안의 `EXPERIMENT_MODE` 값을 원하는 모드로 바꾼 뒤 실행합니다.

```python
EXPERIMENT_MODE = 2
```

예를 들어 Mode 2를 실행하면 `M1`, `M2`, `N_values`, `TRIALS` 설정에 따라 mixed-M adaptive agents의 average success rate를 계산하고 그림을 저장합니다.

## Notes

- 현재 기본 설정은 `TOTAL_STEPS = 2000`, `EQ_STEPS = 0`, `TRIALS = 100`입니다.
- `M1`, `M2`를 바꾸면 저장되는 Figure 2 파일명에도 해당 값이 들어갑니다.
- 계산량은 `N_values`, `TRIALS`, `TOTAL_STEPS`가 커질수록 빠르게 증가합니다.
