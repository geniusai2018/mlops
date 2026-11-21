# 제2부 3장 — 단계를 나누고, 입력이 바뀐 단계만 다시 돌리기 (오케스트레이터·dvc repro 가 해 주는 일의 축소판)
import pathlib as _pl
_D = str(next(c for p in _pl.Path(__file__).resolve().parents for c in (p / "공용데이터", p / "30_실습" / "공용데이터") if c.exists())) + "/"   # 실습 묶음의 공용데이터/ 또는 02_MLOps/30_실습/공용데이터
import hashlib, json, pathlib, sys
import pandas as pd
from sklearn.linear_model import LogisticRegression
W = pathlib.Path("steps"); W.mkdir(exist_ok=True)
SRC = pathlib.Path(_D)
LOCK = W / "lock.json"; lock = json.loads(LOCK.read_text()) if LOCK.exists() else {}
def sha(*paths): 
    h = hashlib.sha256()
    for p in paths: h.update(pathlib.Path(p).read_bytes())
    return h.hexdigest()[:12]
def step(name, deps, outs, fn, params=None):
    key = sha(*deps) + json.dumps(params or {}, sort_keys=True)
    if lock.get(name) == key and all(pathlib.Path(o).exists() for o in outs):
        print(f"  {name:6s} 건너뜀 (입력 그대로)"); return
    fn(); lock[name] = key; print(f"  {name:6s} 실행")
F = ["attend_rate_4w", "late_count_4w"]
def prepare():
    df = pd.concat([pd.read_csv(SRC / f"trainees_{c}.csv") for c in ["2025-1", "2025-2"]]).drop_duplicates()
    df[F] = df[F].fillna(df[F].median()); df.to_csv(W / "clean.csv", index=False)
def train():
    df = pd.read_csv(W / "clean.csv"); C = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
    m = LogisticRegression(C=C, max_iter=1000).fit(df[F], df.dropped)
    (W / "model.json").write_text(json.dumps({"coef": m.coef_.round(4).tolist()}))
step("준비", [SRC / "trainees_2025-1.csv", SRC / "trainees_2025-2.csv"], [W / "clean.csv"], prepare)
step("학습", [W / "clean.csv"], [W / "model.json"], train, {"C": sys.argv[1] if len(sys.argv) > 1 else "1.0"})
LOCK.write_text(json.dumps(lock, ensure_ascii=False, indent=1))
