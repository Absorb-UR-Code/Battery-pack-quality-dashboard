# n8n 불량 로그 수신 워크플로

`01_불량로그_수신_및_보관.json`은 Streamlit이 전송한 불량 원본 CSV와
판정 메타데이터를 받아 Google Drive와 n8n Data Table에 저장한다.

## 가져온 뒤 반드시 설정할 항목

1. `불량로그 Webhook` 노드에서 Header Auth credential을 다시 선택한다.
2. Header 이름과 값은 Streamlit Secrets의 `auth_header_name`,
   `auth_token`과 정확히 같아야 한다.
3. `원본 CSV Google Drive 보관` 노드에서 Google Drive credential과
   `FaultLogs` 폴더를 다시 선택한다.
4. `불량 이벤트 Data Table Upsert` 노드에서 실제 불량 이벤트 테이블을
   다시 선택한다.
5. 저장한 뒤 워크플로를 Active로 전환하고 Production URL
   `/webhook/battery-pack-fault`를 Streamlit Secrets에 입력한다.

## 정상 응답

Google Drive 업로드와 Data Table 저장이 끝나면 webhook은 다음 형태의
JSON을 반환한다.

```json
{
  "ok": true,
  "event_id": "전송한 이벤트 ID",
  "archived": true,
  "stored_at": "저장 시각"
}
```

HTTP 200이더라도 응답 본문이 비어 있으면 저장 완료를 확인할 수 없는
상태다. 이 경우 동일한 webhook 경로를 사용하는 이전 워크플로가 활성화되어
있는지, Data Table 노드가 출력을 반환하지 않고 멈추는지 확인한다.

## Streamlit 전송 시점

실시간 화면의 중간 윈도우에서 불량 표시가 떠도 바로 전송하지 않는다.
선택한 CSV의 마지막 행까지 판정이 끝난 뒤 파일 단위 대표 불량 1건을
확정하고, 그때 원본 CSV 전체와 메타데이터를 전송한다.
