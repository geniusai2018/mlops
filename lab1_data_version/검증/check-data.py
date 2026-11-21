"""데이터 버전 점검.  사용법:  python 검증/check-data.py [작업폴더]

무엇을 보는가 — **되짚을 수 있는가.**
  「이 모델은 무엇으로 배웠습니까」에 답하려면, 그때 그 데이터를 다시 꺼낼 수 있어야 합니다.
  도구(DVC 등)가 대신 해 주는 일을 여기서는 손으로 한 번 만들어 봅니다. 원리는 같습니다 — 해시와 기록.

기대하는 구조
  작업폴더/
    data/v1/trainees.csv  ·  data/v1/MANIFEST.json
    data/v2/trainees.csv  ·  data/v2/MANIFEST.json
    데이터카드.md
"""
import hashlib, json, sys, pathlib

ROOT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "시작코드").resolve()
ok = bad = 0
out = []

def check(cid, cond, detail=""):
    global ok, bad
    if cond: ok += 1
    else: bad += 1
    out.append(f"{'  통과' if cond else '❌ 실패'}  {cid}{'  — ' + detail if detail else ''}")

def sha(p):
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()

data = ROOT / "data"
versions = sorted(d for d in data.glob("v*") if d.is_dir()) if data.is_dir() else []

# D1 — 버전이 둘 이상 있는가
check("D1  데이터 버전이 2개 이상", len(versions) >= 2,
      f"찾은 버전: {[v.name for v in versions] or '없음'}")

# D2 — 버전마다 기록이 있는가
missing = [v.name for v in versions if not (v / "MANIFEST.json").exists()]
check("D2  버전마다 MANIFEST.json 이 있다", versions and not missing,
      f"없는 곳: {missing}" if missing else "")

# D3 — 기록한 해시가 실제 파일과 맞는가
mismatch, checked = [], 0
for v in versions:
    mf = v / "MANIFEST.json"
    if not mf.exists(): continue
    try: man = json.loads(mf.read_text(encoding="utf-8"))
    except Exception: mismatch.append(f"{v.name}:읽기실패"); continue
    f = v / man.get("file", "")
    if not f.exists(): mismatch.append(f"{v.name}:파일없음"); continue
    checked += 1
    if man.get("sha256") != sha(f): mismatch.append(f"{v.name}:해시불일치")
check("D3  기록한 해시가 실제 파일과 맞는다", checked > 0 and not mismatch,
      " · ".join(mismatch) if mismatch else f"{checked}개 확인")

# D4 — 무엇이 들어 있는지 적었는가
need = {"rows", "duplicates", "missing"}
lack = []
for v in versions:
    mf = v / "MANIFEST.json"
    if not mf.exists(): continue
    try: man = json.loads(mf.read_text(encoding="utf-8"))
    except Exception: continue
    miss = need - set(man)
    if miss: lack.append(f"{v.name}:{sorted(miss)}")
check("D4  행수·중복·결측을 기록했다", versions and not lack, " · ".join(lack) if lack else "")

# D5 — 두 버전이 실제로 다른가
hashes = []
for v in versions:
    mf = v / "MANIFEST.json"
    if mf.exists():
        try: hashes.append(json.loads(mf.read_text(encoding="utf-8")).get("sha256"))
        except Exception: pass
check("D5  버전끼리 내용이 다르다", len(set(h for h in hashes if h)) >= 2,
      "해시가 전부 같다 — 버전을 나눈 의미가 없다" if len(set(hashes)) < 2 else "")

# D6 — 사람이 읽을 기록이 있는가
card = ROOT / "데이터카드.md"
body = card.read_text(encoding="utf-8") if card.exists() else ""
check("D6  데이터카드.md 가 있고 버전 차이를 적었다",
      card.exists() and len(body.strip()) > 200 and ("v2" in body or "v1" in body),
      "없거나 너무 짧다" if not (card.exists() and len(body.strip()) > 200) else "")

print(f"\n점검 대상: {ROOT}\n" + "\n".join(out))
print(f"\n통과 {ok} · 실패 {bad}\n")
if bad:
    print("되짚을 수 없는 데이터로 배운 모델은, 나중에 아무것도 설명할 수 없습니다.\n")
sys.exit(1 if bad else 0)
