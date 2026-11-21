"""그 경보는 믿을 만한가 — 심화 시작 코드

    python sensitivity.py        → sensitivity.json

실습 5에서는 PSI 를 구간 10개로 한 번 쟀습니다. 여기서 셋을 봅니다.
  ① 구간 수를 바꾸면 판정이 바뀌는가
  ② **같은 세상**을 둘로 나눠도 경보가 울리는가 (헛경보) — 표본이 작으면 어떻게 되는가
  ③ 다른 자(KS 검정)로 재면 같은 말을 하는가
"""
import json
import pathlib

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp  # noqa: F401 — TODO 3 에서 씁니다

HERE = pathlib.Path(__file__).resolve().parent
SRC = next(p / "공용데이터" for p in HERE.parents if (p / "공용데이터").is_dir())
FEATURES = ["attend_rate_4w", "late_count_4w", "assign_submit_rate", "assign_late_rate", "consult_count", "prior_experience"]
PSI_WATCH, PSI_ALERT = 0.1, 0.2      # 이 과정이 정한 선 (실습 5와 같음)
BINS = [5, 10, 20]
SIZES = [310, 100, 50]               # 새로 들어온 쪽의 표본 수
REPEATS = 300
rng = np.random.default_rng(20261113)


def psi(base, new, bins=10):
    """실습 5의 것과 같습니다."""
    base, new = np.asarray(base, float), np.asarray(new, float)
    edges = np.unique(np.quantile(base, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        cats = np.unique(np.concatenate([base, new]))
        pb = np.array([(base == c).mean() for c in cats])
        pn = np.array([(new == c).mean() for c in cats])
    else:
        edges[0], edges[-1] = -np.inf, np.inf
        pb = np.histogram(base, bins=edges)[0] / len(base)
        pn = np.histogram(new, bins=edges)[0] / len(new)
    pb, pn = np.clip(pb, 1e-4, None), np.clip(pn, 1e-4, None)
    return round(float(((pn - pb) * np.log(pn / pb)).sum()), 4)


base = pd.concat([pd.read_csv(SRC / f"trainees_{c}.csv") for c in ["2025-1", "2025-2"]], ignore_index=True).drop_duplicates()
MED = base[FEATURES].median()
base[FEATURES] = base[FEATURES].fillna(MED)
cohorts = {}
for c in ["2026-1", "2026-2"]:       # 2026-1 은 같은 세상(대면), 2026-2 는 바뀐 세상(비대면)
    d = pd.read_csv(SRC / f"trainees_{c}.csv")
    d[FEATURES] = d[FEATURES].fillna(MED)
    cohorts[c] = d

# ① TODO 1 — 구간 수(BINS)마다 PSI 를 재십시오. 두 기수 × 여섯 칸 × 구간 셋.
#    bins = {"2026-1": {"attend_rate_4w": {"5": …, "10": …, "20": …}, …}, "2026-2": {…}}
bins = {}

# ② TODO 2 — 헛경보. 배운 자료(base)를 아무렇게나 둘로 갈라 PSI 를 잽니다. REPEATS 번.
#    한쪽은 절반(half), 다른 쪽은 n 행(SIZES). 「여섯 칸 중 하나라도」 선을 넘은 비율을 셉니다.
#    false_alarm = {"310": {"any_watch": …, "any_alert": …}, "100": {…}, "50": {…}}
#    섞을 때는 rng.permutation(len(base)) 를 쓰십시오 (씨앗이 고정되어 있습니다).
half = len(base) // 2
false_alarm = {}

# ③ TODO 3 — 같은 비교를 KS 검정으로. ks_2samp(a, b).pvalue
#    ks = {"2026-1": {"attend_rate_4w": …, …}, "2026-2": {…}}
ks = {}

result = {"bins": bins, "repeats": REPEATS, "false_alarm": false_alarm, "ks": ks}
(HERE / "sensitivity.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(result, ensure_ascii=False, indent=2)[:1200])
