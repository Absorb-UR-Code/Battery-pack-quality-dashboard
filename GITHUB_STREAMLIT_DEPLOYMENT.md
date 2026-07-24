# GitHub 및 Streamlit Community Cloud 배포 안내

## 1. GitHub에 올려야 하는 파일

GitHub 저장소의 최상위 화면에 `app.py`가 바로 보이도록 아래 구조를 그대로
업로드합니다. 상위 폴더를 한 번 더 감싸지 않는 것이 중요합니다.

```text
저장소 루트/
├─ app.py
├─ requirements.txt
├─ README.md
├─ GITHUB_STREAMLIT_DEPLOYMENT.md
├─ .gitignore
├─ .streamlit/config.toml
├─ components/
├─ config/settings.json
├─ core/
├─ data/demo/test/
├─ data/inbox/.gitkeep
├─ models/
├─ outputs/.gitkeep
└─ tests/
```

### 반드시 포함

| 구분 | 경로 | 역할 |
|---|---|---|
| 앱 | `app.py` | Streamlit 시작 파일 |
| 핵심 코드 | `core/` | 데이터 탐색, 파생변수, 모델 호출, 로그 및 시각화 |
| UI 컴포넌트 | `components/` | KPI 드래그·배치 작업 영역 |
| 클라우드 설정 | `config/settings.json` | `data/demo`, `data/inbox` 연결 |
| 운영 LSTM | `models/lstm_two_stage_quality_v1/` | 1단계 이진분류와 2단계 유형분류 |
| LSTM-AE | `models/lstm_ae_fault_bank_v1/` | 선택형 재구성 모델 |
| 라이브러리 | `requirements.txt` | 클라우드 Python 패키지 설치 |
| 화면 설정 | `.streamlit/config.toml` | 테마와 Streamlit 설정 |
| 예제 데이터 | `data/demo/test/Test01~09_*.csv` | 배포 직후 화면·모델 확인 |

각 모델 폴더 안의 `.keras`, `.npz`, `adapter.py`, `manifest.json`,
`metadata.json`은 서로 한 묶음입니다. 하나라도 빠지면 해당 모델을 로딩할 수
없으므로 폴더 전체를 올립니다.

## 2. GitHub에 올리지 않는 파일

| 제외 대상 | 이유 |
|---|---|
| `config/settings.local.json` | 개인 PC 절대경로가 포함됨 |
| 전체 Train 원본 약 305MB | 실행 시 필요 없고 저장소 복제·배포가 느려짐 |
| Test `*_Label.csv` | 현재 운영 추론 화면에서 사용하지 않음 |
| `outputs/` 내부 로그 | 실행할 때 생성되는 결과물 |
| `data/inbox/` 업로드 파일 | 실행 중 임시로 들어오는 현장 데이터 |
| `archived_models/` | 운영에 사용하지 않는 중복·보관 모델 |
| `__pycache__/`, `*.log` | Python 캐시와 로컬 실행 로그 |

이 항목들은 `.gitignore`에 등록되어 있습니다.

## 3. 배포용 데이터 구성

### 현재 포함한 온라인 예제 데이터

`data/demo/test/`에는 라벨 파일을 제외한 Test01~09 원본 9개를 둡니다.

| 파일군 | 의미 |
|---|---|
| Test01, Test03 | 정상 충전 |
| Test02, Test04 | 정상 방전 |
| Test05, Test06, Test08 | 불량 충전 |
| Test07, Test09 | 불량 방전 |

총 크기는 약 31MB이며 개별 파일은 약 1.2~6.9MB입니다.

### Train 데이터가 불필요한 이유

현재 사이트는 모델을 새로 학습하지 않고 저장된 Keras 모델로 추론합니다.
전처리 기준도 각 모델 폴더의 `scaler.npz`에 포함되어 있으므로 Train 원본은
필요하지 않습니다. Train은 EDA, 재학습, 모델 재검증을 수행할 때만 로컬에서
사용합니다.

### 새로운 현장 데이터 사용

1. 사이트의 `데이터 현황` 탭을 엽니다.
2. CSV 또는 CSV ZIP을 업로드합니다.
3. 업로드 데이터는 실행 중 `data/inbox`에 저장되어 목록에 표시됩니다.
4. Community Cloud의 로컬 저장소는 영구 저장소가 아니므로 중요한 원본은
   별도로 보관합니다.

## 4. GitHub 저장소 생성과 업로드

데이터와 모델이 포함되므로 우선 **Private 저장소**를 권장합니다.

### 방법 A: GitHub 웹에서 바로 업로드

현재 배포 묶음은 45개 파일이고 가장 큰 개별 파일도 25MB 미만이므로 GitHub
웹 업로드 제한 안에 들어갑니다.

1. GitHub에서 `New repository`를 누릅니다.
2. 이름을 `battery-pack-quality-dashboard`로 입력합니다.
3. 공개 승인이 확인되지 않았다면 `Private`를 선택합니다.
4. README 자동 생성은 선택하지 않고 빈 저장소를 만듭니다.
5. 준비된 `battery_pack_ops_dashboard_github_ready_*.zip`을 먼저 압축 해제합니다.
6. 저장소에서 `uploading an existing file` 또는 `Add file > Upload files`를 누릅니다.
7. **ZIP 파일 자체가 아니라 압축을 푼 폴더 안의 45개 항목**을 모두 끌어 놓습니다.
8. 업로드 화면에서 `app.py`, `.streamlit`, `core`, `models`, `data`가 표시되는지
   확인한 뒤 `Commit changes`를 누릅니다.

GitHub 웹은 한 번에 100개 파일, 개별 파일 25MiB까지 업로드할 수 있어 현재
묶음은 한 번에 업로드할 수 있습니다.

### 방법 B: GitHub Desktop

1. GitHub Desktop을 설치하고 GitHub 계정으로 로그인합니다.
2. `File > Add local repository`를 선택합니다.
3. 로컬 폴더로 `battery_pack_ops_dashboard`를 지정합니다.
4. 저장소가 아직 Git 저장소가 아니면 `create a repository`를 선택합니다.
5. Repository name을 `battery-pack-quality-dashboard`로 지정합니다.
6. 첫 커밋 메시지를 `Initial Streamlit deployment`로 작성하고 Commit합니다.
7. `Publish repository`를 누릅니다.
8. `Keep this code private`를 체크하고 게시합니다.

### 방법 C: Git 명령

Git이 설치된 PC에서 저장소 루트에서 실행합니다.

```powershell
git init
git add .
git commit -m "Initial Streamlit deployment"
git branch -M main
git remote add origin https://github.com/<GitHub-ID>/battery-pack-quality-dashboard.git
git push -u origin main
```

`<GitHub-ID>`는 본인 GitHub 사용자명 또는 조직명으로 바꿉니다.

### 업로드 후 확인

GitHub 저장소 최상위에 다음 항목이 보이면 구조가 맞습니다.

```text
app.py
requirements.txt
core/
models/
data/
.streamlit/
```

`config/settings.local.json`, Train 전체 데이터, `outputs`의 CSV 로그가 보이면
업로드를 중단하고 `.gitignore` 적용 여부를 다시 확인합니다.

## 5. Streamlit Community Cloud에서 실행

1. [Streamlit Community Cloud](https://share.streamlit.io/)에 접속합니다.
2. GitHub 계정과 연결합니다.
3. 우측 상단 `Create app`을 누릅니다.
4. `Yup, I have an app`을 선택합니다.
5. 다음 값을 입력합니다.

| 항목 | 입력값 |
|---|---|
| Repository | `<GitHub-ID>/battery-pack-quality-dashboard` |
| Branch | `main` |
| Main file path | `app.py` |
| App URL | 원하는 영문 주소 |

6. `Advanced settings`를 열고 Python을 `3.11`로 선택합니다.
7. `Deploy`를 누릅니다.
8. 최초 설치 시 TensorFlow 설치와 모델 초기화 때문에 수 분이 걸릴 수 있습니다.

배포 후에는 GitHub의 `main` 브랜치에 변경사항을 push하면 Community Cloud가
자동으로 앱을 갱신합니다.

## 6. 배포 오류별 점검

### `ModuleNotFoundError`

- `requirements.txt`가 저장소 루트에 있는지 확인합니다.
- 오류에 표시된 패키지가 `requirements.txt`에 있는지 확인합니다.
- Python 3.11로 배포했는지 확인합니다.

### 모델 파일을 찾지 못함

- `models/<모델 ID>/manifest.json`이 있는지 확인합니다.
- `.keras`, `.npz`, `adapter.py`를 같은 모델 폴더에 올렸는지 확인합니다.
- GitHub에서 파일 크기가 0B가 아닌지 확인합니다.

### 데이터 목록이 비어 있음

- `data/demo/test/` 아래에 CSV가 있는지 확인합니다.
- `config/settings.json`의 경로가 `data/demo`, `data/inbox`인지 확인합니다.
- Windows 절대경로나 역슬래시 경로를 클라우드 설정에 넣지 않습니다.

### 업로드·검토 로그가 재시작 후 사라짐

이는 Community Cloud의 임시 파일 시스템 특성입니다. 현장 운영 버전에서는
원본 파일과 판정·검토 로그를 외부 DB 또는 객체 저장소에 연결해야 합니다.

## 7. 보안 및 공개 범위

- 데이터와 모델의 외부 공개 승인이 없다면 GitHub 저장소를 Private로 유지합니다.
- Private 저장소를 Streamlit에 연결하려면 저장소 관리자 권한과 GitHub 접근
  승인이 필요합니다.
- `.streamlit/secrets.toml`은 GitHub에 올리지 않습니다. 비밀값은 Community
  Cloud의 `App settings > Secrets`에 입력합니다.

## 8. 공식 참고 문서

- [Streamlit Community Cloud 앱 배포](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)
- [Streamlit 배포 파일 구조](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/file-organization)
- [Streamlit 의존성 관리](https://docs.streamlit.io/deploy/concepts/dependencies)
- [GitHub 저장소 및 파일 크기 제한](https://docs.github.com/en/repositories/creating-and-managing-repositories/repository-limits)
