"""데이터에 버전을 붙인다 — 실습

    python 버전만들기.py
    python ../검증/check-data.py .        ← 6가지를 봅니다

도구(DVC 등)가 대신 해 주는 일을 손으로 한 번 만들어 봅니다. 원리는 셋뿐입니다.
  ① 파일을 그대로 보관한다        ② 내용을 해시로 지문을 찍는다
  ③ 무엇이 들어 있는지 함께 적는다
"""
import hashlib, json, pathlib, datetime
import pandas as pd

HERE = pathlib.Path(__file__).parent
SRC = HERE.parent.parent / "공용데이터"
FEATURES = ["attend_rate_4w", "late_count_4w", "assign_submit_rate",
            "assign_late_rate", "consult_count", "prior_experience"]


def save_version(df, name, note):
    """data/<name>/ 에 파일과 MANIFEST.json 을 남긴다."""
    d = HERE / "data" / name
    d.mkdir(parents=True, exist_ok=True)
    f = d / "trainees.csv"
    df.to_csv(f, index=False, encoding="utf-8")

    # TODO 1 — MANIFEST 를 채우십시오.
    #   점검기가 보는 것: version · file · sha256 · rows · duplicates · missing
    #   해시는 hashlib.sha256(f.read_bytes()).hexdigest()
    manifest = {
        
  "version": "v1",
  "file": "trainees.csv",
  "sha256": "cb6276f2367e8ff8e032b292f781c252e0b96b3dc0cf31a2a49adacaa1a177e1",
  "rows": 632,
  "duplicates": 12,
  "missing": 25,
  "created": "2026-11-15",
  "note": "받은 그대로. 손대지 않음"
 
    }
    (d / "MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{name} 저장")
    return manifest


raw = pd.concat([pd.read_csv(SRC / f"trainees_{c}.csv") for c in ["2025-1", "2025-2"]],
                ignore_index=True)

# TODO 2 — v1 은 받은 그대로 저장하십시오. 손대지 않습니다
save_version(raw, "v1", "받은 그대로. 손대지 않음")

# TODO 3 — v2 는 학습에 쓸 수 있게 손본 것입니다.
#   중복을 어떻게 할지, 결측을 어떻게 할지 **여러분이 정하십시오.**
#   정답은 없습니다. 다만 **무엇을 왜 했는지 데이터카드에 적으십시오.**
clean = raw.drop_duplicates().copy()
clean[FEATURES] = clean[FEATURES].fillna(clean[FEATURES].median())
save_version(clean, "v2", "중복 제거 · 결측은 중앙값으로 채움")
