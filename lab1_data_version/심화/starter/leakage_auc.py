"""칸을 넣었다 뺐다 하며 AUC 를 잰다.   python leakage_auc.py

TODO — SAFE 에 「넣어도 되는 칸」만 남기십시오. 지금은 셋이 다 들어 있습니다.
"""
import pathlib

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 20261121
BASE = ["attend_rate_4w", "late_count_4w", "assign_submit_rate", "assign_late_rate", "consult_count", "prior_experience"]
NEW = ["commute_minutes", "refund_requested", "final_survey_done"]
SAFE = ["commute_minutes", "refund_requested", "final_survey_done"]  # TODO

df = pd.read_csv(pathlib.Path(__file__).resolve().parent.parent / "incoming" / "trainees_2025_extra.csv")
df[BASE] = df[BASE].fillna(df[BASE].median())


def auc(cols):
    X_tr, X_va, y_tr, y_va = train_test_split(df[cols], df["dropped"], test_size=0.25, random_state=SEED, stratify=df["dropped"])
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED))
    return round(float(roc_auc_score(y_va, m.fit(X_tr, y_tr).predict_proba(X_va)[:, 1])), 4)


for name, cols in (("원래 6칸", BASE), ("6칸 + 셋 전부", BASE + NEW), ("6칸 + 넣어도 되는 칸만", BASE + SAFE)):
    print(f"{name:<22} AUC {auc(cols)}")
