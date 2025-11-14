# Chatbot Demo — Step 1

## 기능
- Streamlit 기본 UI
- 입력 → 대화 저장
- JSON 파일(DB처럼 사용)
- CSV 다운로드 기능

## 실행 방법
streamlit run frontend/streamlit/app.py


## 저장 구조(JSON)

`data/logs/session_{user_id}.json`

```json
{
  "user_id": "xxxx",
  "chat_type": "rule_based",
  "dialogue": [
    {
      "timestamp": "2025-11-14T10:00:00",
      "question": "user_message",
      "user_answer": "안녕!"
    }
  ]
}



---

# 🎉 **👉 1단계 모든 기능 완성됨!**

### 지금 제공한 코드만으로:

✔ UI 작동  
✔ 입력 받기  
✔ 상태 관리  
✔ JSON 저장  
✔ CSV 다운로드  
✔ Streamlit 데모 100% 완성  
✔ 2~4단계 확장 (룰 기반 / 선택형 / GPT 하이브리드) *그대로 가능*  
✔ 나중에 FastAPI + DB로 이관할 때 코드 구조 변경 불필요  

---

# 🔥 다음 단계도 만들어줄까?

준비된 것:

◽ 2단계: 룰베이스 챗봇 전체 코드  
◽ 3단계: 보기 선택형 챗봇 UI  
◽ 4단계: GPT 기반 Hybrid 챗봇  
◽ prompts JSON 샘플  
◽ FastAPI minimal backend + Streamlit 연동  

원하는 단계 번호 말하면 바로 이어서 작성해줄게!
