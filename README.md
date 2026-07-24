# 배터리팩 품질보증 AI 대시보드

176개 셀 전압 센서와 32개 모듈 온도 센서가 포함된 배터리팩 충·방전 CSV를
실시간 형태로 재생하고, LSTM 기반 정상·불량 판정과 불량 유형 분석 결과를
제공하는 Streamlit 대시보드입니다.

## 운영 모델

- `lstm_two_stage_quality_v1`
  - 1단계: 정상/불량 이진 판정
  - 2단계: 용량, 용접·접촉, 센싱와이어, 온도 센서 불량 유형 분류
- `lstm_ae_fault_bank_v1`
  - 정상 및 불량 유형별 재구성 오차를 사용하는 검증 후보 모델
  - 운영 화면에서 사용자가 선택할 때만 로딩

모델과 전처리 스케일러는 `models/` 아래에 함께 포함되어 있으므로,
온라인 대시보드 실행에 Train 원본 데이터는 필요하지 않습니다.

## 저장소 구조

```text
battery_pack_ops_dashboard/
├─ app.py
├─ requirements.txt
├─ .gitignore
├─ .streamlit/
│  └─ config.toml
├─ components/
│  └─ kpi_workspace/
├─ config/
│  ├─ settings.json
│  └─ settings.local.json       # 로컬 전용, GitHub 업로드 제외
├─ core/
├─ data/
│  ├─ demo/test/                # 온라인 화면 확인용 Test01~09
│  └─ inbox/                    # 실행 중 업로드한 CSV 임시 저장
├─ models/
│  ├─ lstm_two_stage_quality_v1/
│  └─ lstm_ae_fault_bank_v1/
├─ outputs/                     # 판정·검토 로그 임시 저장
└─ tests/
```

## 로컬 실행

Python 3.11 환경에서 저장소 루트를 열고 실행합니다.

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

브라우저에서 `http://localhost:8501`로 접속합니다.

현재 PC의 전체 Train/Test 데이터는 GitHub에서 제외되는
`config/settings.local.json`에 연결되어 있습니다. 이 파일이 없으면
`config/settings.json`의 `data/demo`와 `data/inbox`를 사용합니다.

## GitHub와 Streamlit Community Cloud

상세 절차는 [GITHUB_STREAMLIT_DEPLOYMENT.md](GITHUB_STREAMLIT_DEPLOYMENT.md)를
참조하십시오.

배포 화면에서 다음 값을 사용합니다.

- Repository: 업로드한 GitHub 저장소
- Branch: `main`
- Main file path: `app.py`
- Python: `3.11`

## 운영상 주의사항

- `data/inbox`에 업로드한 파일과 `outputs`의 판정·검토 로그는 Streamlit
  Community Cloud 재시작 또는 재배포 때 사라질 수 있습니다.
- 현장 운영에서는 판정 로그를 PostgreSQL, Supabase, S3 같은 영구 저장소에
  연결해야 합니다.
- Test05~09 및 파생 증강 데이터가 모델 학습에 사용된 이력이 있으므로,
  현재 성능은 완전히 새로운 팩에 대한 독립 일반화 성능으로 해석하면 안 됩니다.
- 모델 최초 선택 시 TensorFlow와 Keras 모델을 로딩하므로 첫 판정에 시간이
  걸릴 수 있습니다.
