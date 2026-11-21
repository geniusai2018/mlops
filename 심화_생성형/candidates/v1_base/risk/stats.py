"""출결 기록을 숫자로 바꿉니다. 여기에는 AI 가 없습니다 — 계산은 코드가 합니다."""
PRESENT = ("출석", "지각", "조퇴")
EXCLUDED = ("공결",)  # V5 — 공결은 분모에서 뺀다
LEVELS = ("낮음", "주의", "위험")  # 약한 것부터 센 것 순서


def attendance_stats(records: list[dict]) -> dict:
    """records: [{date, period, status, reason}, …]"""
    counts = {s: 0 for s in ("출석", "지각", "조퇴", "결석", "공결")}
    for r in records:
        counts[r["status"]] += 1
    counted = sum(n for s, n in counts.items() if s not in EXCLUDED)
    present = sum(counts[s] for s in PRESENT)
    rate = round(present / counted * 100, 1) if counted else None

    streak = 0  # 가장 최근부터 거슬러 올라가며 이어진 결석 수
    for r in sorted(records, key=lambda r: (r["date"], r["period"]), reverse=True):
        if r["status"] != "결석":
            break
        streak += 1
    return {"counts": counts, "counted": counted, "attendance_rate": rate, "recent_absent_streak": streak}


def rule_level(stats: dict) -> str | None:
    """이 실습이 정한 기준입니다. 자기 기관의 수료 기준에 맞춰 바꾸십시오."""
    rate = stats["attendance_rate"]
    if rate is None:
        return None
    if rate < 80 or stats["recent_absent_streak"] >= 6:
        return "위험"
    if rate < 90 or stats["recent_absent_streak"] >= 3:
        return "주의"
    return "낮음"
