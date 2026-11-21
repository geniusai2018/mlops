"""재현성 점검 — 이 과정의 시그니처.  사용법:  python 검증/check-repro.py [작업폴더]

무엇을 보는가 — **두 번 돌리면 같은 것이 나오는가.**
  재현되지 않는 학습은 실험 관리도, 배포도, 롤백도 의미가 없어집니다.
  「지난주 그 모델」을 다시 만들 수 없으면, 무엇이 좋아졌는지도 말할 수 없습니다.

약속한 것
  python train.py <출력폴더>
    → <출력폴더>/predictions.csv   검증셋에 대한 예측
    → <출력폴더>/RUN.json          씨앗값·설정·데이터 버전·성능 지표
"""
import hashlib, json, os, pathlib, subprocess, sys, tempfile

ROOT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "시작코드").resolve()
TRAIN = ROOT / "train.py"
ok = bad = 0
out = []

def check(cid, cond, why="", info=""):
    """why 는 실패했을 때만, info 는 언제나 보여 준다."""
    global ok, bad
    if cond: ok += 1
    else: bad += 1
    tail = info or ("" if cond else why)
    out.append(f"{'  통과' if cond else '❌ 실패'}  {cid}{'  — ' + tail if tail else ''}")

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None

check("R1  train.py 가 있다", TRAIN.exists(), why=str(TRAIN))
if not TRAIN.exists():
    print("\n" + "\n".join(out)); print(f"\n통과 {ok} · 실패 {bad}\n"); sys.exit(1)

tmp = pathlib.Path(tempfile.mkdtemp(prefix="repro_"))
runs, errs = [], []
for tag in ("a", "b"):
    d = tmp / tag
    r = subprocess.run([sys.executable, str(TRAIN), str(d)], cwd=ROOT,
                       capture_output=True, text=True,
                       encoding="utf-8", errors="replace", env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    if r.returncode != 0:
        errs.append(f"{tag}: {(r.stderr or r.stdout).strip().splitlines()[-1][:120]}")
    runs.append(d)

check("R2  두 번 실행이 모두 성공한다", not errs, why=" · ".join(errs))

pa, pb = (d / "predictions.csv" for d in runs)
check("R3  두 실행의 **예측이 완전히 같다**",
      pa.exists() and pb.exists() and sha(pa) == sha(pb),
      why="예측 파일이 없다" if not (pa.exists() and pb.exists()) else "같은 설정인데 결과가 달라졌다")

ja, jb = (d / "RUN.json" for d in runs)
man = {}
if ja.exists():
    try: man = json.loads(ja.read_text(encoding="utf-8"))
    except Exception: man = {}
check("R4  RUN.json 이 있다", bool(man), why="없거나 읽을 수 없다")

check("R5  **데이터 버전**을 기록했다 (data_sha256 또는 data_version)",
      any(k in man for k in ("data_sha256", "data_version")),
      why="무엇으로 배웠는지가 남지 않는다")
check("R6  **씨앗값과 설정**을 기록했다 (seed · params)",
      "seed" in man and "params" in man,
      why=f"있는 항목: {sorted(man)[:8]}" if man else "RUN.json 이 비어 있다")
check("R7  **성능 지표**를 기록했다 (metrics)", "metrics" in man)

if ja.exists() and jb.exists():
    try:
        a, b = json.loads(ja.read_text(encoding="utf-8")), json.loads(jb.read_text(encoding="utf-8"))
        same = a.get("metrics") == b.get("metrics")
    except Exception:
        same = False
    check("R8  두 실행의 성능 지표가 같다", same, why="지표가 흔들린다 — 어딘가 고정되지 않았다")
else:
    check("R8  두 실행의 성능 지표가 같다", False, why="RUN.json 이 없다")

print(f"\n점검 대상: {ROOT}\n" + "\n".join(out))
print(f"\n통과 {ok} · 실패 {bad}\n")
if bad:
    print("흔한 원인 — 나누기·모델·섞기에 씨앗값이 빠졌거나, 파일을 읽는 순서가 정해져 있지 않습니다.\n")
sys.exit(1 if bad else 0)
