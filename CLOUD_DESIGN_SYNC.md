# Streamlit Cloud 디자인 동기화

이 배포본은 로컬 화면과 Streamlit Community Cloud 화면을 동일하게 유지하도록
밝은 테마를 명시적으로 고정했습니다.

## 반드시 함께 배포할 항목

- `app.py`
- `.streamlit/config.toml`
- `components/`
- `config/`
- `core/`
- `data/`
- `models/`
- `requirements.txt`

특히 `.streamlit/config.toml`이 빠지면 데이터 표, 입력창, 보조 버튼이 브라우저의
어두운 테마를 따라갈 수 있습니다. GitHub 저장소에도 숨김 폴더를 포함한 전체
구조를 그대로 반영해야 합니다.

## 고정된 테마

```toml
[theme]
base = "light"
primaryColor = "#0F766E"
backgroundColor = "#F3F6F5"
secondaryBackgroundColor = "#FFFFFF"
textColor = "#172220"
font = "sans serif"
```

배포 후 Streamlit Community Cloud에서 **Manage app > Reboot app**을 실행하면
새 설정으로 다시 시작됩니다.
