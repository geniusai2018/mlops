"""추론 API 계약 점검.  사용법:  python 검증/contract-test.py [작업폴더]

무엇을 보는가 — **부르는 쪽이 믿어도 되는 약속을 지키는가.**
  모델이 좋은지는 묻지 않습니다. 약속은 모델이 바뀌어도 그대로여야 합니다.
  그래야 다음 주에 모델을 갈아 끼워도 부르는 쪽이 안 깨집니다.

약속
  GET  /health    200 · {"status": ...}
  GET  /model     200 · 이름과 버전을 밝힌다
  POST /predict   200 · risk(0~1) · label(0|1) · version
                  잘못된 입력에는 4xx — **500은 실패다**
"""
import importlib.util, json, pathlib, sys, warnings
warnings.filterwarnings("ignore")

ROOT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "시작코드").resolve()
SRC = ROOT / "serve.py"
ok = bad = 0
out = []

def check(cid, cond, why="", info=""):
    """why 는 실패했을 때만, info 는 언제나 보여 준다."""
    global ok, bad
    if cond: ok += 1
    else: bad += 1
    tail = info or ("" if cond else why)
    out.append(f"{'  통과' if cond else '❌ 실패'}  {cid}{'  — ' + tail if tail else ''}")

def report(code=None):
    print(f"\n점검 대상: {ROOT}\n" + "\n".join(out))
    print(f"\n통과 {ok} · 실패 {bad}\n")
    if bad:
        print("약속이 흔들리면 모델이 아무리 좋아도 쓸 수 없습니다.\n")
    sys.exit(1 if bad else 0)

check("C0  serve.py 가 있다", SRC.exists(), why=f"{SRC} 가 없다")
if not SRC.exists(): report()

# ── 불러오기 ──────────────────────────────────────────────
sys.path.insert(0, str(ROOT))
app = None
err = ""
try:
    spec = importlib.util.spec_from_file_location("serve", SRC)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    app = getattr(mod, "app", None)
except Exception as e:
    err = f"{type(e).__name__}: {e}"

check("C1  불러올 수 있고 app 이 있다", app is not None,
      why=err or "app 이라는 이름의 FastAPI 객체가 없다")
if app is None: report()

from fastapi.testclient import TestClient
c = TestClient(app, raise_server_exceptions=False)

GOOD = {"attend_rate_4w": 0.62, "late_count_4w": 5, "assign_submit_rate": 0.41,
        "assign_late_rate": 0.55, "consult_count": 0, "prior_experience": 0}

# ── C2 살아 있는가 ────────────────────────────────────────
r = c.get("/health")
check("C2  GET /health 가 200 을 준다", r.status_code == 200,
      why=f"{r.status_code}", info="200")

# ── C3 지금 누가 답하는가 ──────────────────────────────────
r = c.get("/model")
body = r.json() if r.status_code == 200 else {}
has = r.status_code == 200 and any("name" in k or "model" in k for k in body) and \
      any("version" in k for k in body)
check("C3  GET /model 이 이름과 버전을 밝힌다", has,
      why=f"{r.status_code} · {json.dumps(body, ensure_ascii=False)[:80]}",
      info=json.dumps(body, ensure_ascii=False)[:60] if has else "")

# ── C4 답을 주는가 ───────────────────────────────────────
r = c.post("/predict", json=GOOD)
p = r.json() if r.status_code == 200 else {}
shape = (r.status_code == 200 and isinstance(p.get("risk"), (int, float))
         and 0.0 <= p["risk"] <= 1.0 and p.get("label") in (0, 1))
check("C4  POST /predict 가 risk(0~1) 와 label(0|1) 을 준다", shape,
      why=f"{r.status_code} · {json.dumps(p, ensure_ascii=False)[:80]}",
      info=f"risk={p.get('risk')} label={p.get('label')}" if shape else "")

# ── C5 답과 함께 누가 답했는지 ──────────────────────────────
check("C5  답에 모델 버전이 같이 온다", bool(str(p.get("version", "")).strip()),
      why="나중에 「그때 그 답」을 추적할 끈이 없다",
      info=f"v{p.get('version')}" if p.get("version") else "")

# ── C6 두 번 물으면 같은 답 ────────────────────────────────
r2 = c.post("/predict", json=GOOD)
same = r2.status_code == 200 and r2.json().get("risk") == p.get("risk")
check("C6  같은 입력에 같은 답을 준다", same,
      why="같은 것을 물었는데 답이 다르다")

# ── C7 잘못된 입력을 막는가 ────────────────────────────────
missing = dict(GOOD); missing.pop("attend_rate_4w")
r = c.post("/predict", json=missing)
check("C7  항목이 빠지면 4xx 로 막는다", 400 <= r.status_code < 500,
      why=f"{r.status_code} — 500 이면 막은 것이 아니라 터진 것이다",
      info=str(r.status_code))

outrange = dict(GOOD); outrange["attend_rate_4w"] = 1.5
r = c.post("/predict", json=outrange)
check("C8  범위를 벗어난 값을 4xx 로 막는다", 400 <= r.status_code < 500,
      why=f"출석률 1.5 를 {r.status_code} 로 받았다 — 있을 수 없는 값이다",
      info=str(r.status_code))

# ── C9 버전을 박지 않았는가 ────────────────────────────────
import re
body_src = SRC.read_text(encoding="utf-8")
alias_used = bool(re.search(r"models:/[^\"']*@", body_src)) or \
             "get_model_version_by_alias" in body_src
hardcoded = bool(re.search(r"models:/[^\"']+/\s*\d", body_src))
check("C9  모델을 **별칭**으로 부른다 (버전 번호를 박지 않았다)",
      alias_used and not hardcoded,
      why="버전을 박으면 갈아 끼울 때마다 이 파일을 고쳐야 한다")

report()
