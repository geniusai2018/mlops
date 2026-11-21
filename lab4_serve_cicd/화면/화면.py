"""방금 만든 것을 화면으로 확인하기 — 실습 4 · 실습 6 마지막 단계

    (실습 폴더의 시작코드 안에서)
    python ../화면/화면.py               → 브라우저가 저절로 열립니다  http://127.0.0.1:8000/

  · 실습 4  여러분의 serve.py 를 그대로 띄우고, 훈련생 한 명을 넣어 위험도를 그림으로 봅니다.
  · 실습 6  여기에 더해, 여러분의 promote.py · rollback.py 를 버튼으로 눌러 봅니다.
            관문을 아직 안 세웠다면 불량 모델도 그냥 올라갑니다 — 그것도 화면에 그대로 보입니다.

이 파일은 여러분 코드를 **고치지 않고 불러다 쓰기만** 합니다. 점검기 결과와도 상관없습니다.
끄기: 이 창에서 Ctrl+C
"""
import contextlib
import importlib.util
import io
import json
import os
import sys
import threading
import traceback
from pathlib import Path

for _s in (sys.stdout, sys.stderr):  # 한글 윈도우(cp949)에서 출력을 파일로 돌려도 멈추지 않게
    try:
        _s.reconfigure(errors="replace")
    except Exception:
        pass

for k, v in {"MLFLOW_DISABLE_AGENT_HINT": "1", "MLFLOW_DISABLE_TELEMETRY": "true", "DO_NOT_TRACK": "1",
             "PYTHONWARNINGS": "ignore"}.items():
    os.environ.setdefault(k, v)
import warnings  # noqa: E402
warnings.filterwarnings("ignore")
import logging  # noqa: E402
logging.getLogger("mlflow").setLevel(logging.ERROR)

HERE = Path(__file__).resolve().parent
CWD = Path.cwd()
MODEL_NAME, ALIAS = "dropout-risk", "production"


def stop(msg: str):
    raise SystemExit("\n" + msg + "\n")


if not (CWD / "mlflow.db").exists():
    stop("이 폴더에 mlflow.db 가 없습니다.\n"
         "  · 실습 폴더의 시작코드(또는 완성본) 안에서 실행하십시오:  cd 시작코드  →  python ../화면/화면.py\n"
         "  · 처음이라면 먼저:  python 준비.py")

SERVE = CWD / "serve.py" if (CWD / "serve.py").exists() else CWD.parent / "공용" / "serve.py"
if not SERVE.exists():
    stop("serve.py 를 찾지 못했습니다 (이 폴더 또는 ../공용/).")
LAB6 = (CWD / "promote.py").exists() and (CWD / "rollback.py").exists()
LOG = CWD / "승격이력.jsonl"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


try:
    serve = _load("serve", SERVE)
except Exception:  # 수강생 코드가 아직 덜 됐을 수 있습니다 — 무엇이 막혔는지 그대로 보여 줍니다
    traceback.print_exc()
    stop(f"{SERVE.name} 를 불러오지 못했습니다(위 오류). 먼저 점검기를 돌려 보십시오:  python ../검증/…")

from fastapi import HTTPException  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
import mlflow  # noqa: E402
from mlflow import MlflowClient  # noqa: E402

mlflow.set_tracking_uri(f"sqlite:///{(CWD / 'mlflow.db').as_posix()}")
client = MlflowClient()
app = serve.app
LOCK = threading.Lock()
CACHE = {}
promote = rollback = None
if LAB6:
    promote = _load("promote", CWD / "promote.py")
    rollback = _load("rollback", CWD / "rollback.py")


def _alias():
    try:
        return str(client.get_model_version_by_alias(MODEL_NAME, ALIAS).version)
    except Exception:
        return None


def _switch():
    """실습 6: 별칭이 옮겨 갔으면 serve.py 가 답하는 모델도 따라 바꿉니다 (서버를 다시 띄우지 않고)."""
    if not all(hasattr(serve, a) for a in ("mv", "MODEL", "USES")):
        return
    mv = client.get_model_version_by_alias(MODEL_NAME, ALIAS)
    v = str(mv.version)
    if v not in CACHE:
        CACHE[v] = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{v}")
    serve.mv, serve.MODEL = mv, CACHE[v]
    serve.USES = (mv.tags.get("features") or ",".join(serve.ALL_FEATURES)).split(",")


def _history():
    if not LOG.exists():
        return []
    out = []
    for line in LOG.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            pass
    return out


def _run(mod, *args):
    out, old = io.StringIO(), sys.argv
    sys.argv = [mod.__name__ + ".py", *args]
    code = 0
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            code = mod.main() or 0
    except SystemExit as e:
        code = e.code or 0
    except Exception:
        out.write(traceback.format_exc(limit=3))
        code = "오류"
    finally:
        sys.argv = old
    return out.getvalue().strip(), code


def _state():
    cur = _alias()
    gates = {str(h.get("to")): h["gate"] for h in _history() if h.get("gate")}
    vers = []
    for mv in sorted(client.search_model_versions(f"name='{MODEL_NAME}'"), key=lambda m: int(m.version)):
        feats = (mv.tags.get("features") or "").split(",") if mv.tags.get("features") else []
        label = mv.tags.get("label", "")
        if not label:
            try:
                label = client.get_run(mv.run_id).info.run_name or ""
            except Exception:
                label = ""
        vers.append({"version": str(mv.version), "label": label,
                     "features": len(feats) or None, "production": str(mv.version) == cur,
                     "gate": gates.get(str(mv.version))})
    floor = {"recall": getattr(promote, "RECALL_FLOOR", 0.6), "precision": getattr(promote, "PRECISION_FLOOR", 0.4)}
    label = next((v["label"] for v in vers if v["production"]), "")
    return {"mode": "lab6" if LAB6 else "lab4", "current": cur, "label": label, "versions": vers,
            "floor": floor, "history": _history()}


def _act(mod, *args):
    if not LAB6:
        raise HTTPException(404, "모델 바꿔 끼우기는 실습 6에서 합니다.")
    with LOCK:
        before, n = _alias(), len(_history())
        text, code = _run(mod, *args)
        try:
            _switch()
        except Exception:
            pass
        new = _history()[n:]
        return {"result": new[-1] if new else None, "before": before, "after": _alias(),
                "code": code, "text": text, "state": _state()}


@app.get("/demo/state", include_in_schema=False)
def demo_state():
    return _state()


@app.post("/demo/promote/{version}", include_in_schema=False)
def demo_promote(version: str):
    return _act(promote, version, "화면에서 올려 봄")


@app.post("/demo/rollback", include_in_schema=False)
def demo_rollback():
    return _act(rollback)


@app.get("/", include_in_schema=False)
def page():
    return FileResponse(HERE / "화면.html")


try:
    _switch()
except Exception:
    pass


def main():
    import socket
    import webbrowser
    import uvicorn
    port = int(sys.argv[sys.argv.index("--port") + 1]) if "--port" in sys.argv else 8000
    with socket.socket() as s:
        if s.connect_ex(("127.0.0.1", port)) == 0:
            stop(f"{port}번을 이미 다른 프로그램이 쓰고 있습니다 — 앞서 띄운 uvicorn 창이 있으면 Ctrl+C 로 끄고 다시 실행하십시오.\n"
                 f"  또는:  python ../화면/화면.py --port {port + 1}")
    url = f"http://127.0.0.1:{port}/"
    line = "━" * 58
    what = "실습 6 — 예측 + 올려 보기 · 되돌리기" if LAB6 else "실습 4 — 훈련생 한 명 넣어 보기"
    print(f"\n{line}\n  화면이 떴습니다 ({what})\n  → {url}\n"
          f"  브라우저가 저절로 열립니다. 안 열리면 위 주소를 주소창에 붙여 넣으십시오.\n"
          f"  이 창은 서버입니다 — 닫거나 Ctrl+C 를 누르면 화면도 멈춥니다.\n{line}\n", flush=True)
    if "--no-browser" not in sys.argv:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    main()
