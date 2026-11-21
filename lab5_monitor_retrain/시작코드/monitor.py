"""드리프트 감시와 재학습 판단 (실습)

    python 준비.py                       ← 처음 한 번만
    python monitor.py
    python ../검증/check-drift.py .

새 기수(2026-2) 자료가 들어왔습니다. 비대면으로 바뀐 기수입니다.
이 스크립트가 답해야 할 질문은 셋입니다.

  ① 들어오는 값이 예전과 다른가        입력 분포 (PSI)
  ② 답이 예전만큼 맞는가              운영 모델 성능
  ③ 그래서 다시 배워야 하는가          판단

지금은 ①의 **숫자만 내고 판정을 안 합니다.** ②와 ③은 비어 있습니다.

할 일은 다섯입니다.
  ① 판정한다        PSI 값에 기준을 대어 안정 / 주의 / 이상으로
  ② 성능을 잰다      옛 기수에서와 새 기수에서, 둘 다
  ③ 판단한다        무엇이 걸렸을 때 다시 배우기로 하는가 — **근거를 남긴다**
  ④ 다시 배운다      세 가지 방법을 견주어 하나를 고르고 **등록**한다
  ⑤ 손대지 않는다    **production 별칭은 그대로 둔다.** 승격은 사람이 한다

마지막 줄이 이 실습의 핵심입니다. 자동으로 올리면 사고도 자동이 됩니다.
"""
import json, pathlib
import numpy as np
import pandas as pd
import mlflow
from mlflow import MlflowClient
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score

SEED = 20261121
HERE = pathlib.Path(__file__).resolve().parent
SRC = HERE.parent.parent / "공용데이터"
DB = HERE / "mlflow.db"
MODEL_NAME = "dropout-risk"
ALIAS = "production"

FEATURES = ["attend_rate_4w", "late_count_4w", "assign_submit_rate",
            "assign_late_rate", "consult_count", "prior_experience"]
TARGET = "dropped"
BASE_COHORTS = ["2025-1", "2025-2"]      # 모델이 배운 세상
NEW_COHORT = "2026-2"                    # 새로 들어온 세상

# 이 과정이 정한 선입니다. 절대 기준이 아니라 **약속**입니다. (흔히 인용되는 관례는 0.1·0.25)
#   0.1 미만 안정 · 0.1~0.2 주의 · 0.2 이상 이상
PSI_WATCH, PSI_ALERT = 0.1, 0.2
RECALL_FLOOR = 0.60                      # lab3 실험비교.md 에 적어 둔 하한


def psi(base, new, bins=10):
    """두 분포가 얼마나 벌어졌는가. 0에 가까우면 그대로, 커지면 딴 세상.
    (이 함수는 만들어 두었습니다. 고치지 않아도 됩니다)"""
    base, new = np.asarray(base, float), np.asarray(new, float)
    edges = np.unique(np.quantile(base, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:                                   # 값 종류가 적은 열 (0/1 같은)
        cats = np.unique(np.concatenate([base, new]))
        pb = np.array([(base == c).mean() for c in cats])
        pn = np.array([(new == c).mean() for c in cats])
    else:
        edges[0], edges[-1] = -np.inf, np.inf
        pb = np.histogram(base, bins=edges)[0] / len(base)
        pn = np.histogram(new, bins=edges)[0] / len(new)
    pb, pn = np.clip(pb, 1e-4, None), np.clip(pn, 1e-4, None)
    return round(float(((pn - pb) * np.log(pn / pb)).sum()), 4)


def evaluate(model, X, y, uses):
    """성능 네 가지를 한 번에. (이것도 만들어 두었습니다)"""
    pred = model.predict(X[uses])
    proba = model.predict_proba(X[uses])[:, 1]
    return {"recall": round(float(recall_score(y, pred)), 4),
            "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
            "f1": round(float(f1_score(y, pred)), 4),
            "auc": round(float(roc_auc_score(y, proba)), 4)}


# ── 자료 (손대지 않아도 됩니다) ───────────────────────────────
base = pd.concat([pd.read_csv(SRC / f"trainees_{c}.csv") for c in BASE_COHORTS],
                 ignore_index=True).drop_duplicates()
MED = base[FEATURES].median()          # 결측 채우는 기준은 **학습 때 것**을 쓴다
base[FEATURES] = base[FEATURES].fillna(MED)

new = pd.read_csv(SRC / f"trainees_{NEW_COHORT}.csv")
new[FEATURES] = new[FEATURES].fillna(MED)

mlflow.set_tracking_uri(f"sqlite:///{DB.as_posix()}")
client = MlflowClient()
mv = client.get_model_version_by_alias(MODEL_NAME, ALIAS)
prod = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{ALIAS}")
USES = (mv.tags.get("features") or ",".join(FEATURES)).split(",")

# 새 기수를 학습분과 평가분으로 나눕니다. 평가분은 **새 모델도 못 본 자료**여야
# 비교가 공정합니다.
Xn_tr, Xn_va, yn_tr, yn_va = train_test_split(
    new[FEATURES], new[TARGET], test_size=0.25, random_state=SEED, stratify=new[TARGET])

# ── ① 입력 분포 ───────────────────────────────────────────
cols, drifted, watched = {}, [], []
for f in FEATURES:
    v = psi(base[f], new[f])
    # ── TODO 1 ─ 판정하십시오 ─────────────────────────────
    # 지금은 숫자만 적고 상태를 "미판정"으로 둡니다.
    # PSI_ALERT 이상이면 "이상", PSI_WATCH 이상이면 "주의", 아니면 "안정".
    # 이상인 열은 drifted 에, 주의인 열은 watched 에 담으십시오.
    cols[f] = {"baseline_mean": round(float(base[f].mean()), 4),
               "target_mean": round(float(new[f].mean()), 4),
               "psi": v, "state": "미판정", "drift": False}

# ── ② 성능 ───────────────────────────────────────────────
# ── TODO 2 ─ 운영 모델 성능을 두 자리에서 재십시오 ──────────────
#   baseline_perf = 옛 기수(base) 에서   → evaluate(prod, base[FEATURES], base[TARGET], USES)
#   before        = 새 기수 평가분에서    → evaluate(prod, Xn_va, yn_va, USES)
baseline_perf, before = {}, {}

# ── ③ 판단 ───────────────────────────────────────────────
# ── TODO 3 ─ 규칙을 쓰고 **근거를 남기십시오** ────────────────
# 걸릴 만한 것 둘: 입력 분포에 이상이 있다 / 재현율이 하한 아래로 내려갔다
# 「왜 다시 배우기로 했는가」가 reasons 에 문장으로 남아야 합니다.
reasons = []
retrain = bool(reasons)

# ── ④ 재학습 ──────────────────────────────────────────────
# ── TODO 4 ─ 세 가지를 견주어 하나를 고르고 등록하십시오 ────────
# **재학습은 「다시 돌리기」가 아니라 「무엇으로 배울지 다시 정하기」입니다.**
#   전부_섞기    옛 기수 전부 + 새 기수 학습분
#   새기수_3배   옛 기수 전부 + 새 기수 학습분을 세 번
#   새기수만     새 기수 학습분만
# 셋 다 학습해 **새 기수 평가분(Xn_va, yn_va)** 으로 견주고, 재현율로 고릅니다.
# 고른 것을 mlflow 에 기록하고 registered_model_name 으로 등록하십시오.
# 버전 태그에 "trigger"(왜 다시 배웠나) 를 꼭 남기십시오 — 점검기가 봅니다.
after, new_version, trials, best_label = None, None, [], None

# ── ⑤ 별칭은 건드리지 않습니다 ───────────────────────────────
# ⚠ 여기에 set_registered_model_alias 를 부르고 싶어집니다. **부르지 마십시오.**
#    무엇을 올릴지는 사람이 정합니다 (lab6).
still = client.get_model_version_by_alias(MODEL_NAME, ALIAS).version

report = {
    "baseline": {"cohorts": BASE_COHORTS, "rows": int(len(base)), "performance": baseline_perf},
    "target": {"cohort": NEW_COHORT, "rows": int(len(new))},
    "features": cols,
    "drifted": drifted,
    "watched": watched,
    "performance": {"model_version": mv.version, "on_baseline": baseline_perf,
                    "on_target": before},
    "decision": {"retrain": retrain, "reasons": reasons, "recall_floor": RECALL_FLOOR},
    "retrained": ({"version": new_version, "strategy": best_label, "trials": trials,
                   "metrics_on_target": after,
                   "promoted": False} if retrain else None),
    "production_alias_after": still,
}
(HERE / "drift_report.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

print(f"\n입력 분포  이상 {drifted or '없음'} · 주의 {watched or '없음'}")
print(f"판단      재학습 {'필요' if retrain else '불필요'}")
print(f"별칭      production = v{still}\n")
