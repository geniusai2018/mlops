"""요약 한 건의 모양과 **내용의 한계** — 모델이 준 답은 이 문을 지나야 화면에 나갑니다."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Level = Literal["낮음", "주의", "위험"]


class RiskSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")  # 모르는 칸이 끼어 오면 거절

    level: Level
    attendance_rate: float = Field(ge=0, le=100)
    summary: str = Field(min_length=10, max_length=200)
    evidence: list[str] = Field(min_length=1, max_length=3)
