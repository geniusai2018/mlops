"""고정 사례 평가 — 후보 하나를 표본 20명 × 가짜 모델 다섯에 돌려 본다. (완성되어 있습니다 · 고치지 않습니다)

    from evaluation import evaluate, fingerprint
    evaluate(후보폴더) → {"cases": 100, "failed": n, "leaks": n, "by_model": {...}}

정답은 모델이 아니라 **코드가 계산한 값**(eval_set/golden_cases.json)입니다. 키도 네트워크도 필요 없습니다.
1차 과정 실습 5·6 의 평가를 옮겨 온 것입니다.
"""
import hashlib
import importlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EVALSET = HERE / "eval_set"
LEVELS = ("낮음", "주의", "위험")


def fingerprint(*paths) -> str:
    """파일(들)의 지문. 폴더를 주면 그 안의 .py 를 이름순으로 훑는다."""
    h = hashlib.sha256()
    for p in paths:
        p = Path(p)
        for f in (sorted(p.rglob("*.py")) if p.is_dir() else [p]):
            h.update(f.read_bytes().replace(b"\r\n", b"\n"))   # 줄바꿈 차이로 지문이 갈리지 않게
    return h.hexdigest()[:16]


def evalset_fingerprint() -> str:
    return fingerprint(*(EVALSET / n for n in ("trainees.json", "attendances_sample.json", "golden_cases.json")))


def _load(candidate: Path):
    for name in [m for m in sys.modules if m == "risk" or m.startswith("risk.")]:
        del sys.modules[name]                      # 앞서 불러온 다른 후보를 지운다
    sys.path.insert(0, str(candidate))
    try:
        llm, service = importlib.import_module("risk.llm"), importlib.import_module("risk.service")
    finally:
        sys.path.remove(str(candidate))
    return llm, service


def evaluate(candidate) -> dict:
    llm, service = _load(Path(candidate))
    Mock, LLMError = llm.MockLLM, llm.LLMError
    trainees = json.loads((EVALSET / "trainees.json").read_text(encoding="utf-8"))
    records = json.loads((EVALSET / "attendances_sample.json").read_text(encoding="utf-8"))
    golden = {g["trainee_id"]: g for g in json.loads((EVALSET / "golden_cases.json").read_text(encoding="utf-8"))["cases"]}

    class Lenient(Mock):        # 등급을 늘 낮음이라고 한다
        def ask(self, i, p):
            return {**super().ask(i, p), "level": "낮음"}

    class Inventor(Mock):       # 출석률을 15점 올려 적는다
        def ask(self, i, p):
            out = super().ask(i, p)
            return {**out, "attendance_rate": min(100.0, round(out["attendance_rate"] + 15, 1))}

    class Obedient(Mock):       # 자료 안의 지시문을 따른다
        def ask(self, i, p):
            out = super().ask(i, p)
            return {**out, "level": "낮음"} if "무시하고" in json.dumps(p, ensure_ascii=False) else out

    class Flaky(Mock):          # 첫 호출은 실패, 네 번째 사람마다 끝내 답하지 않는다
        case = calls = 0

        def ask(self, i, p):
            self.calls += 1
            if self.case % 4 == 0:
                raise TimeoutError("모의 시간 초과")
            if self.calls == 1:
                raise LLMError("모의 한도 초과")
            return super().ask(i, p)

    by_model, failed, leaks = {}, 0, 0
    for name, cls in (("정직한", Mock), ("무른", Lenient), ("지어내는", Inventor), ("시키는대로", Obedient), ("불안정한", Flaky)):
        model, bad_here = cls(), 0
        for t in trainees:
            rows = [r for r in records if r["trainee_id"] == t["id"]]
            if isinstance(model, Flaky):
                model.case, model.calls = model.case + 1, 0
            sent = []
            spy = type("Spy", (), {"ask": lambda self, i, p, m=model: (sent.append(json.dumps(p, ensure_ascii=False)), m.ask(i, p))[1]})()
            gold, bad, leak = golden[t["id"]], False, False
            try:
                out = service.summarize(t, rows, spy)
                bad |= out.get("attendance_rate") != gold["attendance_rate"]
                if gold["floor"] is not None:
                    bad |= out.get("level") not in LEVELS or LEVELS.index(out["level"]) < LEVELS.index(gold["floor"])
                bad |= out.get("source") not in ("llm", "rule", "none")
            except Exception:
                bad = True
            text = " ".join(sent)
            leak = t["name"] in text or t["contact_last4"] in text or any(r.get("reason") and r["reason"] in text for r in rows)
            bad_here += bad or leak
            leaks += leak
        by_model[name] = len(trainees) - bad_here
        failed += bad_here
    return {"cases": 5 * len(trainees), "failed": int(failed), "leaks": int(leaks), "by_model": by_model}
