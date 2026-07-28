# n8n Webhook 연동

대시보드가 새로운 불량 이벤트를 `outputs/fault/model_fault_event_log.csv`에
기록한 직후, **방금 불량 판정을 받은 개별 원본 CSV 파일 전체**를 n8n
Webhook으로 전송한다. 누적 불량 로그 파일 자체는 전송하지 않는다.

## 동작 원칙

- CSV 저장을 먼저 완료한 뒤 Webhook을 호출한다.
- 전송 형식은 `multipart/form-data`이며 파일 필드명은 `fault_source_csv`이다.
- 첨부 파일명은 불량 판정을 받은 원본 CSV 이름을 그대로 사용한다.
- 첨부파일에는 해당 원본 CSV의 모든 행과 모든 열이 포함된다.
- 다른 원본 CSV에서 새 불량이 발생하면 해당 파일만 별도로 전송한다.
- 같은 `event_id`가 Streamlit 재실행으로 다시 저장되어도 n8n에는 다시 보내지 않는다.
- Webhook 오류나 시간 초과가 발생해도 모델 판정과 CSV 저장은 계속된다.
- 전송 결과는 로컬 CSV의 `n8n_delivery_status`, `n8n_delivery_at`,
  `n8n_http_status`, `n8n_delivery_error` 열에 남는다.

## 함께 전송되는 값

Multipart 요청에는 다음 항목이 포함된다.

| 필드 | 내용 |
| --- | --- |
| `fault_source_csv` | 이번에 불량 판정을 받은 원본 CSV 전체 파일 |
| `metadata` | 행 수, 파일 크기, SHA-256, 트리거 이벤트 정보가 담긴 JSON 문자열 |
| `event_id` | 이번 전송을 발생시킨 불량 이벤트 ID |
| `row_count` | CSV 전체 데이터 행 수 |

HTTP 헤더에는 `X-Battery-Event-Id`와 `X-Battery-Log-SHA256`도 함께 들어간다.
n8n에서 `event_id` 또는 SHA-256을 기준으로 추가 중복 방지를 적용할 수 있다.

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

Webhook 실행 데이터에서 CSV는 Binary 데이터의 `fault_source_csv` 항목으로 확인한다.
후속 노드에서는 이 Binary 데이터를 Google Drive, S3, DB 또는 메일 첨부파일로
전달할 수 있다.

## 주의사항

- 판정 대상 원본 CSV 전체를 보내므로 파일에 저장된 모든 센서 행과 열이 외부로 전달된다.
- 동일 원본 파일에서 서로 다른 신규 불량 이벤트가 발생하면 각각 별도의 전송이
  발생할 수 있다. 동일한 `event_id`만 중복 전송에서 제외된다.
- n8n Webhook의 허용 Payload 크기를 넘으면 HTTP 오류가 기록되지만 대시보드 판정과
  로컬 CSV 저장은 중단되지 않는다.
