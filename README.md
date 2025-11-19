# Chatbot Demo — Step 2
### 5문항 고정 질문 챗봇

아동 상담 캐릭터 **봉봉(BongBong)**이 고정된 5개의 질문을 순서대로 제시하고  
사용자는 차례대로 답변하는 구조의 **룰베이스 챗봇 데모**입니다.

---

## ✨ 주요 기능

- **5문항 고정 질문 자동 진행**
  - 첫 접속 시 질문 1 자동 출력
  - 답변하면 다음 질문 자동 진행
  - 5문항 모두 끝나면 입력창 숨김

- **말풍선 대화 UI(Chat Bubble)**
  - 왼쪽: 봇(봉봉)
  - 오른쪽: 사용자
  - Step 1과 동일한 UI 구성

- **실시간 저장(JSONL)**
  - 메시지 1건마다 즉시 저장
  - bot + user = 1턴 기준으로 turn 구성

- **대화 저장 기능**
  - 5문항 완료 후 "대화 저장" 버튼 활성화
  - JSON / CSV 다운로드 지원

- **GPT 코드 포함(미사용)**
  - 프로젝트 구조 통일을 위해 GPT 모듈 유지
  - 데모2는 GPT 응답 생성 없이 룰베이스로 동작

---

## 📁 디렉토리 구조

```text
chatbot-demo-2/
├─ frontend/
│  └─ streamlit/
│       └─ app.py
├─ data/
│  └─ logs/
│       └─ chat_log.jsonl
├─ .env
├─ requirements.txt
└─ README.md
```


🚀 실행 방법

1) 가상환경 활성화 후 실행


```bash
# 가상 환경 생성 (venv 사용)
python -m venv venv

# 가상 환경 활성화 (Linux/macOS)
source venv/bin/activate

# 가상 환경 활성화 (Windows)
.\venv\Scripts\activate

# 라이브러리 설치(의존성)
pip install -r requirements.txt

# 프로젝트 루트 디렉토리에서 Streamlit 앱을 실행
```bash
streamlit run frontend/streamlit/app.py
```

또는 `streamlit` 폴더에서:

```bash
streamlit run app.py
```

***

🧩 챗봇 질문 흐름

데모2에서는 아래 5개 문항이 순서대로 자동 진행됩니다.

1.  그림 속 너는 지금 무엇을 하고 있어?
2.  그림 속 너는 지금 어떤 기분이야?
3.  오늘은 왜 이렇게 그리고 싶었어?
4.  그림 속 너에게 해주고 싶은 말은 뭐야?
5.  내일의 너는 어떤 모습이면 좋겠어?

***

🗄 저장 구조

1) 실시간 저장(JSONL)

경로: `data/logs/chat_log.jsonl`

예시:

```jsonl
{"session_id":"sess_001","timestamp":"2025-11-19T08:24:35","role":"bot","text":"그림 속 너는 지금 무엇을 하고 있어?","turn":1}
{"session_id":"sess_001","timestamp":"2025-11-19T08:24:50","role":"user","text":"나는 놀고 있어!","turn":1}
```

2) JSON 다운로드 구조(다운로드 파일)

예시(`chat_history.json`): 

```json
{
  "session_id": "sess_001",
  "user_id": "user_abc123",
  "created_at": "2025-11-19T10:00:00",
  "chat_type": "rule_based",
  "dialogue": [
    {
      "turn": 1,
      "bot": {"role": "bot", "text": "그림 속 너는 지금 무엇을 하고 있어?", "timestamp": "..."},
      "user": {"role": "user", "text": "나는 놀고 있어!", "timestamp": "..."}
    }
    // ... 나머지 턴
  ]
}
```

3) CSV 다운로드 구조(다운로드 파일))

| session_id | user_id | created_at | chat_type | turn | role | text | timestamp |
|---|---|---|---|---|---|---|---|
| sess_001 | user_abc123 | 2025-11-19T10:00:00 | rule_based | 1 | bot | 그림 속 너는 지금 무엇을 하고 있어? | ... |
| sess_001 | user_abc123 | 2025-11-19T10:00:00 | rule_based | 1 | user | 나는 놀고 있어! | ... |
| ... | ... | ... | ... | ... | ... | ... | ... |