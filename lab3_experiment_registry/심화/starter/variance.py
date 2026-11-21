"""그 차이는 우연인가 — 심화 시작 코드

    python variance.py        → variance.json

실습 3에서는 나누기 한 번으로 B(0.68)와 C(0.64)를 견줬습니다. 나누는 씨앗만 바꿔 스무 번 재어 봅니다.
"""
import json
import pathlib
import statistics  # noqa: F401 — TODO 에서 씁니다

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 20261121
HERE = pathlib.Path(__file__).resolve().parent
SRC = HERE.parents[2] / "공용데이터"
ALL = ["attend_rate_4w", "late_count_4w", "assign_submit_rate", "assign_late_rate", "consult_count", "prior_experience"]
RECIPES = {
    "A_기본": {"features": ALL, "class_weight": None},
    "B_가중치": {"features": ALL, "class_weight": "balanced"},
    "C_출석만": {"features": ALL[:2], "class_weight": "balanced"},
}
SEEDS = list(range(1, 21))          # 나누는 씨앗 스무 개
COST_FN, COST_FP = 10, 1            # 놓친 한 명은 잘못 걸린 열 명 값 — 실습 3의 「두 실수의 값이 다르다」

df = pd.concat([pd.read_csv(SRC / f"trainees_{c}.csv") for c in ["2025-1", "2025-2"]], ignore_index=True).drop_duplicates()
df[ALL] = df[ALL].fillna(df[ALL].median())


def fit_split(cfg, split_seed):
    """split_seed 로 나누고 학습한다. (검증셋 정답, 검증셋 이탈 확률) 을 돌려준다."""
    X_tr, X_va, y_tr, y_va = train_test_split(df[cfg["features"]], df["dropped"], test_size=0.25,
                                              random_state=split_seed, stratify=df["dropped"])
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight=cfg["class_weight"], random_state=SEED))
    return y_va.to_numpy(), model.fit(X_tr, y_tr).predict_proba(X_va)[:, 1]


def scores(y, proba, threshold=0.5):
    pred = (proba >= threshold).astype(int)
    return float(recall_score(y, pred)), float(precision_score(y, pred, zero_division=0))


# ① TODO 1 — 설정마다 SEEDS 로 스무 번 재어, 재현율·정밀도의 평균과 표준편차를 구하십시오.
#    recipes = {"A_기본": {"recall_mean": …, "recall_std": …, "precision_mean": …, "precision_std": …}, …}
recipes = {}

# ② TODO 2 — B 와 C 를 **같은 나누기끼리** 짝지어 재현율 차이(B − C)를 구하십시오.
#    b_minus_c = {"recall_diff_mean": …, "recall_diff_std": …, "b_wins": …, "ties": …, "c_wins": …}
b_minus_c = {}

# ③ TODO 3 — 자르는 선(임계값)을 0.05 ~ 0.95 에서 0.05 씩 옮기며, 비용이 가장 작은 선을 찾으십시오.
#    비용 = COST_FN × 놓친 수 + COST_FP × 잘못 걸린 수.  나누기는 SEED, 설정은 B.
#    threshold = {"cost_fn": 10, "cost_fp": 1, "best_threshold": …, "cost_at_best": …,
#                 "cost_at_0.5": …, "recall_at_best": …, "precision_at_best": …}
threshold = {}

result = {"seeds": SEEDS, "recipes": recipes, "B_minus_C": b_minus_c, "threshold": threshold}
(HERE / "variance.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(result, ensure_ascii=False, indent=2))
