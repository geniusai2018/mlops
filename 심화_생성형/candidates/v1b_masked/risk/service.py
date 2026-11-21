"""결석 위험 요약 — 에이전트가 만들어 준 판본입니다. 정상 응답일 때는 잘 돕니다.

실습 TODO 2 — build_payload: 모델에게 **무엇을 보내지 않을지** 정하십시오.
실습 TODO 3 — summarize: 응답이 틀렸을 때, 늦을 때, 실패했을 때를 채우십시오.
"""
from .llm import LLM
from .schema import RiskSummary
from .stats import attendance_stats

INSTRUCTIONS = (
    "너는 직업훈련 과정의 출결 담당자를 돕는다. 받은 출결 자료만 근거로 결석 위험을 요약한다. "
    "level 은 낮음·주의·위험 중 하나, summary 는 한국어 두 문장 이내, evidence 는 근거 1~3개."
)


def build_payload(trainee: dict, records: list[dict]) -> dict:
    # v1 에서 고친 것 하나 — 이름·연락처·사유 원문을 보내지 않는다. 그 밖의 방어는 아직 없다
    recent = sorted(records, key=lambda r: (r["date"], r["period"]))[-12:]
    return {"cohort": trainee["cohort"], "stats": attendance_stats(records),
            "recent": [{"date": r["date"], "status": r["status"], "has_reason": bool(r.get("reason"))} for r in recent]}


def summarize(trainee: dict, records: list[dict], llm: LLM) -> dict:
    raw = llm.ask(INSTRUCTIONS, build_payload(trainee, records))
    s = RiskSummary.model_validate(raw)
    return {"trainee_id": trainee["id"], **s.model_dump(), "source": "llm"}
