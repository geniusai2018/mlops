"""생성형 기능에도 같은 틀을 — 기록 · 비교 · 관문 · 되돌리기 (심화 시작 코드)

    python llmops.py record v1_base          후보를 평가하고 기록한다
    python llmops.py record v2_guarded
    python llmops.py promote v2_guarded "방어 코드 추가"     관문을 통과해야 운영 후보가 된다
    python llmops.py rollback                직전으로
    python llmops.py status

실습 1~6 에서 모델에 하던 일을, **지시문 + 방어 코드 + 평가 세트**의 묶음에 그대로 합니다.
"""
import datetime
import json
import os
import pathlib
import sys

for k, v in (("MLFLOW_DISABLE_TELEMETRY", "true"), ("DO_NOT_TRACK", "1"), ("MLFLOW_DISABLE_AGENT_HINT", "1")):
    os.environ.setdefault(k, v)
import mlflow  # noqa: E402

LAB = next(p for p in pathlib.Path(__file__).resolve().parents if (p / "evaluation.py").exists())
sys.path.insert(0, str(LAB))
from evaluation import evalset_fingerprint, evaluate, fingerprint  # noqa: E402

STORE = pathlib.Path(os.environ.get("LLMOPS_STORE", pathlib.Path(__file__).resolve().parent))
CURRENT, LOG = STORE / "current_candidate.json", STORE / "promotion_log.jsonl"
MODEL = os.environ.get("LLM_MODEL", "gpt-6-luna")
mlflow.set_tracking_uri(f"sqlite:///{(STORE / 'mlflow.db').as_posix()}")
mlflow.set_experiment("결석위험요약_후보")


def instructions_of(candidate: pathlib.Path) -> str:
    sys.modules.pop("risk.service", None)
    sys.modules.pop("risk", None)
    sys.path.insert(0, str(candidate))
    try:
        import risk.service as s
        return s.INSTRUCTIONS
    finally:
        sys.path.remove(str(candidate))


def record(name: str) -> int:
    cand = LAB / "candidates" / name
    result = evaluate(cand)
    with mlflow.start_run(run_name=name):
        # 무엇을 평가했는지 — 넷이 다 있어야 「그때 그 후보」를 다시 가리킬 수 있다
        # TODO 1 — 「그때 그 후보」를 다시 가리킬 수 있게 다섯을 남기십시오.
        #   candidate(이름) · prompt_sha(지시문의 지문) · code_sha(후보 폴더의 지문)
        #   evalset_sha(평가 세트의 지문) · model(모델 이름)
        #   도구: fingerprint_text(글) · fingerprint(폴더) · evalset_fingerprint() · instructions_of(cand) · MODEL
        mlflow.log_params({"candidate": name})
        mlflow.log_metrics({"cases": result["cases"], "failed": result["failed"], "leaks": result["leaks"],
                            **{f"pass_{k}": v for k, v in result["by_model"].items()}})
        mlflow.log_text(instructions_of(cand), "지시문.txt")
    print(f"{name}: {result['cases'] - result['failed']}/{result['cases']} 통과 · 누출 {result['leaks']}건 · {result['by_model']}")
    return 0


def fingerprint_text(text: str) -> str:
    import hashlib
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def latest_run(name: str):
    runs = mlflow.search_runs(filter_string=f"params.candidate = '{name}'", order_by=["attributes.start_time DESC"], max_results=1)
    return None if runs.empty else runs.iloc[0]


def write_log(entry: dict) -> None:
    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def current() -> dict | None:
    return json.loads(CURRENT.read_text(encoding="utf-8")) if CURRENT.exists() else None


def promote(name: str, why: str) -> int:
    run = latest_run(name)
    if run is None:
        print(f"{name} 의 평가 기록이 없습니다. 먼저 record 하십시오.")
        return 2
    fails = []
    # ── 관문 ── 실습 6과 같은 자리입니다. 미달이면 올라가지 않습니다
    # TODO 2 — 셋 중 하나라도 걸리면 fails 에 이유를 넣으십시오.
    #   · 통과하지 못한 사례가 있다            run["metrics.failed"]
    #   · 개인정보·사유 원문이 새어 나갔다      run["metrics.leaks"]
    #   · 평가 세트가 그때와 다르다            run["params.evalset_sha"] 와 지금의 evalset_fingerprint()
    before = current()
    now = datetime.datetime.now().isoformat(timespec="seconds")
    if fails:
        write_log({"at": now, "action": "rejected", "candidate": name, "why": why, "fails": fails})
        print("거부 — " + " · ".join(fails))
        return 1
    # TODO 3 — 이력 한 줄에 from(직전 후보) · to · why · run_id · prompt_sha · evalset_sha 를 담으십시오.
    entry = {"at": now, "action": "promote", "to": name}
    write_log(entry)                                   # 옮기기 전에 직전 자리를 적어 둔다
    CURRENT.write_text(json.dumps({"candidate": name, "run_id": run["run_id"], "since": now}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"승격 — {entry.get('from')} → {name}")
    return 0


def rollback() -> int:
    lines = [json.loads(x) for x in LOG.read_text(encoding="utf-8").splitlines()] if LOG.exists() else []
    last = next((x for x in reversed(lines) if x["action"] in ("promote", "rollback")), None)
    if not last or not last.get("from"):
        print("되돌아갈 자리가 없습니다.")
        return 1
    now = datetime.datetime.now().isoformat(timespec="seconds")
    write_log({"at": now, "action": "rollback", "from": last["to"], "to": last["from"], "why": "되돌리기"})
    CURRENT.write_text(json.dumps({"candidate": last["from"], "since": now}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"되돌림 — {last['to']} → {last['from']}")
    return 0


def main() -> int:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "record" and len(sys.argv) > 2:
        return record(sys.argv[2])
    if cmd == "promote" and len(sys.argv) > 2:
        return promote(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "(사유 없음)")
    if cmd == "rollback":
        return rollback()
    print(json.dumps(current(), ensure_ascii=False) if current() else "운영 후보가 아직 없습니다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
