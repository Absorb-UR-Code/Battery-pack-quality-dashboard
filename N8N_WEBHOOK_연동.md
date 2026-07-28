# n8n Webhook 연동

대시보드가 새로운 불량 이벤트를 `outputs/fault/model_fault_event_log.csv`에
저장한 직후 동일 이벤트를 n8n Webhook으로 전송한다.

## 동작 원칙

- CSV 저장을 먼저 완료한 뒤 Webhook을 호출한다.
- Webhook 오류나 시간 초과가 발생해도 모델 판정과 CSV 저장은 계속된다.
- 같은 `event_id`가 Streamlit 재실행으로 다시 저장되어도 n8n에는 한 번만 보낸다.
- 전송 결과는 CSV의 `n8n_delivery_status`, `n8n_delivery_at`,
  `n8n_http_status`, `n8n_delivery_error` 열에 남는다.
- 로컬 파일 경로인 `source_path`는 외부로 전송하지 않는다.

## Streamlit Community Cloud Secrets

앱의 **Manage app > Settings > Secrets**에 실제 값으로 아래 항목을 추가한다.

```toml
[n8n]
enabled = true
webhook_url = "https://example.app.n8n.cloud/webhook/battery-pack-fault"
auth_header_name = "X-Battery-Token"
auth_token = "32자 이상의 충분히 긴 임의 토큰"
timeout_seconds = 5
send_raw_window = true
```

`webhook_url`에는 n8n Webhook 노드의 **Production URL**을 사용한다.
토큰은 GitHub 파일이나 코드에 기록하지 않는다.

## n8n Webhook 노드

1. HTTP Method를 `POST`로 설정한다.
2. Path를 `battery-pack-fault`로 설정한다.
3. Authentication은 `Header Auth`를 선택한다.
4. Header Name을 Streamlit Secrets의 `auth_header_name`과 같게 설정한다.
5. Header Value를 Streamlit Secrets의 `auth_token`과 같게 설정한다.
6. 워크플로를 저장하고 활성화한 뒤 Production URL을 Secrets에 입력한다.

## 원본 윈도우 전송

- `send_raw_window = true`: 판정에 사용한 원본 센서 윈도우와 현재 행을 전송한다.
- `send_raw_window = false`: 요약·판정·조치 정보만 보내고 원본 센서 윈도우와
  `raw__*` 열은 제외한다.

원본 윈도우는 Payload가 커질 수 있으므로 n8n에서 필요한 열만 DB나 알림 시스템에
저장하는 방식을 권장한다.
