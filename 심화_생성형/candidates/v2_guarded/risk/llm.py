"""LLM 을 부르는 얇은 층. 바깥(service.py)은 「물으면 dict 가 온다」만 압니다.

    LLM_MODE=mock    (기본) 키 없이 돕니다. 규칙으로 답을 지어내는 흉내입니다
    LLM_MODE=openai  진짜로 부릅니다. 환경 변수 OPENAI_API_KEY 가 필요합니다
    LLM_MODEL        모델 이름 (기본 gpt-6-luna)

키는 코드에도, 저장소의 파일에도 적지 않습니다. 환경 변수로만 줍니다.
"""
import json
import os
from typing import Literal, Protocol

from pydantic import BaseModel


class LLMError(Exception):
    """LLM 쪽에서 난 실패 (연결 · 한도 · 거절 · 빈 응답). 다시 해 볼 만한 실패입니다."""


class LLM(Protocol):
    def ask(self, instructions: str, payload: dict) -> dict: ...


class WireSummary(BaseModel):
    """모델에게 요구하는 응답의 **모양**. 길이·범위 같은 내용 검증은 schema.py 가 따로 합니다."""

    level: Literal["낮음", "주의", "위험"]
    attendance_rate: float
    summary: str
    evidence: list[str]


class MockLLM:
    """키 없이 실습하기 위한 흉내. 보낸 숫자를 그대로 읽어 답합니다."""

    def ask(self, instructions: str, payload: dict) -> dict:
        s = payload["stats"]
        rate, c = s["attendance_rate"], s["counts"]
        level = "위험" if rate < 80 else "주의" if rate < 90 else "낮음"
        return {
            "level": level,
            "attendance_rate": rate,
            "summary": f"출석률 {rate}% 로 위험도는 {level}입니다. 결석 {c['결석']}회, 지각·조퇴 {c['지각'] + c['조퇴']}회입니다.",
            "evidence": [f"출석률 {rate}%", f"결석 {c['결석']}회"],
        }


class OpenAILLM:
    def __init__(self, model: str | None = None, timeout: float = 20.0):
        from openai import OpenAI  # 모의 모드에서는 이 패키지가 없어도 됩니다

        self.client = OpenAI(timeout=timeout, max_retries=0)  # 재시도는 service.py 가 정한다
        self.model = model or os.environ.get("LLM_MODEL", "gpt-6-luna")

    def ask(self, instructions: str, payload: dict) -> dict:
        import openai

        try:
            res = self.client.responses.parse(
                model=self.model,
                instructions=instructions,
                input=json.dumps(payload, ensure_ascii=False),
                text_format=WireSummary,
                max_output_tokens=600,
            )
        except openai.APITimeoutError as e:
            raise TimeoutError(str(e)) from e
        except openai.OpenAIError as e:
            raise LLMError(type(e).__name__) from e  # 원문 메시지를 밖으로 흘리지 않는다
        if res.output_parsed is None:
            raise LLMError("empty_or_refused")
        return res.output_parsed.model_dump()


def default_llm() -> LLM:
    return OpenAILLM() if os.environ.get("LLM_MODE", "mock") == "openai" else MockLLM()
