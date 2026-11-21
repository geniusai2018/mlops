# 교재 예시 코드 (2차 MLOps 수강생 교재 v3.3)

교재 본문의 「실행 결과」 상자마다 적힌 파일입니다. 수업 시간표에는 들어 있지 않습니다. 궁금한 절을 읽다가 직접 돌려 보는 용도입니다.

| 파일 | 교재 자리 | 필요한 것 |
|---|---|---|
| e_lineage.py | 1부 2-2 전문가 관점 · 데이터 계보 | mlflow |
| e_nondet.py | 1부 3-3 전문가 관점 · 씨앗값으로 막히지 않는 흔들림 | — |
| e_mcnemar.py | 1부 4-3 전문가 관점 · 차이의 검정 (값은 「실습 후 해설」 8번 H-3) | scipy |
| e_signature.py | 1부 4-6 전문가 관점 · 모델 서명 | mlflow |
| e_throughput.py | 1부 5-2 전문가 관점 · 지연과 처리량 | 실습 4 심화 완성본 `serve.py` 옆(mlflow.db 가 있는 폴더)에서 실행 |
| e_drift.py | 1부 6-3 · 6-4 전문가 관점 · 검정과 데이터셋 이동 | scipy |
| e_steps.py | 제2부 3-4 · 바뀐 단계만 다시 돌리기 | — |
| e_search.py | 제2부 4-4 · 기록 꺼내 비교하기 | mlflow |
| e_predrate.py | 제2부 9-5 · 위험 판정 비율 | — |
| e_weight.py | 제2부 10-4 · 새 자료에 무게 주기 | — |

- 실습 준비 안내대로 깐 환경(pandas · scikit-learn · mlflow · fastapi · uvicorn · httpx · joblib)이면 모두 돌아갑니다. scipy 는 scikit-learn 을 깔 때 함께 깔립니다.
- 이 폴더로 들어가 `python e_drift.py` 처럼 돌립니다. 자료는 위쪽 폴더의 `공용데이터/` 를 찾아 읽습니다.
- 값은 2026-11-13 리눅스 작업 공간에서 잰 것입니다(교재 부록 B). 응답 시간은 장비마다 다릅니다. 윈도우에서는 아직 돌려 보지 않았습니다.
- `e_mcnemar.py` · `e_drift.py` · `e_predrate.py` 는 실습 3 · 5의 결과를 미리 보여 줍니다. 실습을 마친 뒤 돌리십시오.
