"""데이터 계약 — 「이 자료는 이렇게 생겨야 한다」를 코드로 적는다.

    python contract.py ../../완성본/data/v1/trainees.csv        (자기 v1·v2 가 있으면 그 경로)

데이터카드는 사람이 읽는 약속이고, 계약은 **기계가 매번 대어 보는 약속**입니다.
TODO 1 — 계약(CONTRACT)을 채웁니다.   TODO 2 — 범위·허용값·유일성 검사를 채웁니다.
"""
import sys

import pandas as pd

# 칸 이름 → 약속. kind 는 "str" · "int" · "float". 나머지는 필요한 칸에만 적습니다.
#   min · max      값의 범위
#   allowed        들어올 수 있는 값의 목록
#   nullable       비어 있어도 되는가 (기본 False)
#   unique         겹치면 안 되는가 (기본 False)
CONTRACT = {
    "trainee_id": {"kind": "str", "unique": True},
    "cohort": {"kind": "str"},
    # TODO 1 — 나머지 일곱 칸을 적으십시오. 공용데이터/README.md 의 「열」 표를 보십시오.
    #   비율은 0~1, 횟수는 0 이상, 있다/없다는 0·1 입니다.
}


def violations(df: pd.DataFrame, contract: dict = CONTRACT) -> list[str]:
    """계약을 어긴 곳을 문장으로 돌려준다. 빈 목록이면 지킨 것."""
    out = []
    for col in contract:
        if col not in df.columns:
            out.append(f"{col}: 칸이 없다")
    for col in df.columns:
        if col not in contract:
            out.append(f"{col}: 계약에 없는 칸이 들어왔다")

    for col, rule in contract.items():
        if col not in df.columns:
            continue
        s = df[col]
        if not rule.get("nullable", False) and s.isna().any():
            out.append(f"{col}: 빈 값 {int(s.isna().sum())}건")
        # TODO 2 — 아래 셋을 채우십시오. 빈 값은 건너뛰고 봅니다(s.dropna()).
        #   · rule 에 min/max 가 있으면 벗어난 값이 몇 건인지
        #   · rule 에 allowed 가 있으면 목록 밖의 값이 몇 건인지
        #   · rule 에 unique 가 있으면 겹친 값이 몇 건인지
    return out


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "../../완성본/data/v1/trainees.csv"
    found = violations(pd.read_csv(path))
    print(f"{path} — 위반 {len(found)}건")
    for line in found:
        print("  ·", line)
    sys.exit(1 if found else 0)  # 어기면 0 이 아닌 값으로 끝난다 — 자동 점검에 걸 수 있게
