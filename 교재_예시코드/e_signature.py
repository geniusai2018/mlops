# 4장 — 모델 서명: 저장할 때 입력 모양을 같이 적어 두면, 부를 때 어긋난 입력을 막는다
import pathlib as _pl
_D = str(next(c for p in _pl.Path(__file__).resolve().parents for c in (p / "공용데이터", p / "30_실습" / "공용데이터") if c.exists())) + "/"   # 실습 묶음의 공용데이터/ 또는 02_MLOps/30_실습/공용데이터
import warnings; warnings.filterwarnings("ignore")
import pandas as pd, mlflow
from mlflow.models import infer_signature
from sklearn.linear_model import LogisticRegression
SRC=_D + ""
F=["attend_rate_4w","late_count_4w"]
df=pd.read_csv(SRC+"trainees_2025-1.csv").dropna(subset=F)
m=LogisticRegression(max_iter=1000).fit(df[F],df["dropped"])
sig=infer_signature(df[F],m.predict(df[F]))
print(sig)
mlflow.set_tracking_uri("sqlite:///sig/mlflow.db")
with mlflow.start_run():
    info=mlflow.sklearn.log_model(m,name="model",signature=sig)
pm=mlflow.pyfunc.load_model(info.model_uri)
def 물어보기(이름, d):
    try: print(이름, "→", pm.predict(d))
    except Exception as e:
        m=str(e); i=m.find("Error:"); print(이름, "→", (m[i:] if i>=0 else m).splitlines()[0][:100])
물어보기("① 맞는 입력               ", pd.DataFrame({"attend_rate_4w":[0.9],"late_count_4w":[1]}))
물어보기("② 지각 칸이 빠진 입력     ", pd.DataFrame({"attend_rate_4w":[0.9]}))
물어보기("③ 출석률에 글자가 들어옴  ", pd.DataFrame({"attend_rate_4w":["높음"],"late_count_4w":[1]}))
물어보기("④ 지각 횟수에 1.5가 들어옴", pd.DataFrame({"attend_rate_4w":[0.9],"late_count_4w":[1.5]}))
물어보기("⑤ 쓰지 않는 칸이 하나 더  ", pd.DataFrame({"attend_rate_4w":[0.9],"late_count_4w":[1],"memo":["x"]}))
