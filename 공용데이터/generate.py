"""중도탈락 위험 예측 — 가상 데이터 생성기 (강사용)

    python generate.py

기수 4개를 만든다. 전부 생성된 가상 데이터이며 실제 훈련생 정보가 아니다.
씨앗값이 고정되어 있으므로 **몇 번 돌려도 같은 파일**이 나온다 — 이 과정의 주제(재현성)와 같은 원칙이다.

심어 둔 것
  2025-1 · 2025-2 · 2026-1 : 대면 과정. 출석률과 지각이 이탈을 예측한다
  2026-2                    : 비대면 전환. 출석률이 전반적으로 높아지고 지각 개념이 사라진다
                              그래서 대면 데이터로 배운 모델의 성능이 떨어진다
  2025-2 에는 중복 행, 모든 기수에 약간의 결측
"""
import numpy as np, pandas as pd, pathlib

OUT = pathlib.Path(__file__).parent
RNG = np.random.default_rng(20261121)

BASE_ONSITE = -1.27   # 대면 기수의 이탈률 약 0.22
BASE_ONLINE = -0.75   # 비대면 기수의 이탈률 약 0.32 — 비대면에서 이탈이 늘었다

COLS = ["trainee_id", "cohort", "attend_rate_4w", "late_count_4w",
        "assign_submit_rate", "assign_late_rate", "consult_count",
        "prior_experience", "dropped"]


def make(cohort, n, online, start_id):
    """online=False 대면 / online=True 비대면(2026-2)"""
    if not online:
        attend = np.clip(RNG.beta(8, 2, n), 0, 1)          # 평균 0.80 · 퍼져 있다
        late = RNG.poisson(2.2, n)
        submit = np.clip(RNG.beta(7, 2.2, n), 0, 1)
    else:
        attend = np.clip(RNG.beta(30, 2, n), 0, 1)         # 평균 0.94 · 몰려 있다
        late = RNG.poisson(0.05, n)                        # 지각 개념이 사라진다
        submit = np.clip(RNG.beta(6, 2.4, n), 0, 1)
    late_rate = np.clip(RNG.beta(2, 5, n), 0, 1)
    consult = RNG.poisson(0.8, n)
    prior = RNG.binomial(1, 0.35, n)

    # 이탈 확률 — 대면과 비대면에서 "무엇이 신호인지"가 다르다.
    # 모든 항을 표본 평균으로 중심화한다. 그래야 기수별 이탈률이 절편으로만 정해져
    # 기수끼리 비교할 수 있다 (노이즈로 기수 이탈률이 들쭉날쭉하면 드리프트가 가려진다).
    c = lambda v: v - np.mean(v)
    if not online:
        # 대면: 출석률과 지각이 가장 큰 신호
        z = (BASE_ONSITE - 9.0 * c(attend) - 5.0 * c(submit)
             + 0.35 * c(late) + 0.60 * c(late_rate)
             - 0.50 * c(prior) - 0.25 * c(consult))
    else:
        # 비대면: 출석률이 모두 높아 신호를 잃고, 지각 개념이 사라진다. 제출률과 상담이 남는다
        z = (BASE_ONLINE - 6.5 * c(submit) + 1.00 * c(late_rate)
             - 0.50 * c(prior) - 0.50 * c(consult))
    p = 1 / (1 + np.exp(-z))
    dropped = RNG.binomial(1, p)

    df = pd.DataFrame({
        "trainee_id": [f"T{start_id + i:04d}" for i in range(n)],
        "cohort": cohort,
        "attend_rate_4w": attend.round(3),
        "late_count_4w": late,
        "assign_submit_rate": submit.round(3),
        "assign_late_rate": late_rate.round(3),
        "consult_count": consult,
        "prior_experience": prior,
        "dropped": dropped,
    })[COLS]

    # 결측 — 어느 기수에나 조금씩 있다
    for col, k in [("assign_late_rate", max(1, n // 40)), ("consult_count", max(1, n // 60))]:
        idx = RNG.choice(n, k, replace=False)
        df.loc[idx, col] = np.nan
    return df


frames, sid = [], 1
for cohort, n, online in [("2025-1", 300, False), ("2025-2", 320, False),
                          ("2026-1", 310, False), ("2026-2", 320, True)]:
    df = make(cohort, n, online, sid)
    sid += n
    if cohort == "2025-2":                       # 중복 행 8건 — 블록 2의 재료
        df = pd.concat([df, df.sample(12, random_state=7)], ignore_index=True)
    frames.append(df)
    df.to_csv(OUT / f"trainees_{cohort}.csv", index=False, encoding="utf-8")

allf = pd.concat(frames, ignore_index=True)
allf.to_csv(OUT / "trainees_all.csv", index=False, encoding="utf-8")

print("기수별 요약")
print(allf.groupby("cohort").agg(
    n=("trainee_id", "size"),
     출석률=("attend_rate_4w", "mean"),
    지각=("late_count_4w", "mean"),
    제출률=("assign_submit_rate", "mean"),
    이탈률=("dropped", "mean"),
).round(3).to_string())
print(f"\n중복 행: {allf.duplicated().sum()} 건 · 결측: {allf.isna().sum().sum()} 칸")
