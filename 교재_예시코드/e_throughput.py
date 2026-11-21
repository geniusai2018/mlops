# 5장 — 지연(한 건이 얼마나 걸리나)과 처리량(1초에 몇 건을 답하나)
# 실습 4 심화 완성본 serve.py 옆(mlflow.db 가 있는 폴더)에서:  python e_throughput.py
import importlib.util, pathlib, statistics, time, warnings
warnings.filterwarnings("ignore")
from fastapi.testclient import TestClient
spec=importlib.util.spec_from_file_location("serve", pathlib.Path("serve.py")); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
api=TestClient(mod.app)
one={"attend_rate_4w":0.62,"late_count_4w":4,"assign_submit_rate":0.55,"assign_late_rate":0.4,"consult_count":1,"prior_experience":0}
rows=[{**one,"attend_rate_4w":round(0.4+0.003*k,3),"late_count_4w":k%6} for k in range(200)]
for x in rows[:10]: api.post("/predict",json=x)          # 데우기
t=[]; s0=time.perf_counter()
for x in rows:
    a=time.perf_counter(); api.post("/predict",json=x); t.append((time.perf_counter()-a)*1000)
single=time.perf_counter()-s0
q=statistics.quantiles(t,n=100)
b0=time.perf_counter(); api.post("/predict/batch",json={"items":rows}); batch=time.perf_counter()-b0
print(f"한 건씩 200번: 평균 {statistics.mean(t):.1f}ms · p50 {q[49]:.1f}ms · p95 {q[94]:.1f}ms · 걸린 시간 {single:.2f}초 → 처리량 {200/single:.0f}건/초")
print(f"200건 묶음 한 번: {batch*1000:.1f}ms → 처리량 {200/batch:.0f}건/초  (한 건씩보다 {single/batch:.0f}배)")
