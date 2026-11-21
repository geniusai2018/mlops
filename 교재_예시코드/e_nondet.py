# 3장 — 씨앗값을 고정해도 남는 흔들림: 더하는 순서와 문자열 해시
import random, subprocess, sys
import numpy as np
print("(0.1 + 0.2) + 0.3 =", (0.1+0.2)+0.3, "| 0.1 + (0.2 + 0.3) =", 0.1+(0.2+0.3))
rng=random.Random(0); xs=[rng.random() for _ in range(100000)]; ys=xs[:]; random.Random(1).shuffle(ys)
def 차례로(v):
    t=0.0
    for x in v: t+=x
    return t
a,b=차례로(xs),차례로(ys)
print("같은 수 10만 개를 한 줄로 더함 :", repr(a), "| 순서만 섞어 더함 :", repr(b), "| 차이", a-b)
code="print(list({'출석','지각','결석','조퇴','공결'}))"
for s in ["1","2","1"]:
    out=subprocess.run([sys.executable,"-c",code],env={"PYTHONHASHSEED":s},capture_output=True,text=True).stdout.strip()
    print(f"PYTHONHASHSEED={s} → 집합을 목록으로:", out)
