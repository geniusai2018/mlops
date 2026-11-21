"""요약 한 건의 모양과 **내용의 한계**.

실습 TODO 1 — 지금은 아무 값이나 통과합니다. 한계를 적으십시오.
    · level 은 낮음·주의·위험 셋 중 하나
    · attendance_rate 는 0~100
    · summary 는 10자 이상 200자 이하
    · evidence 는 1~3개
    · 모르는 칸이 끼어 오면 거절
"""
from pydantic import BaseModel


class RiskSummary(BaseModel):
    level: str
    attendance_rate: float
    summary: str
    evidence: list[str]
