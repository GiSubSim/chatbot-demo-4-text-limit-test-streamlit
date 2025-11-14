import os
import sys
import json
from datetime import datetime
import streamlit as st
from dotenv import load_dotenv
import pandas as pd

# 환경 변수 로드
load_dotenv()

# 프로젝트 루트 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

# OpenAI
from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -------------------------------------------------
# GPT-4o 챗봇 응답 함수 # 이후 프롬프트 불러오기로 변경하기
# -------------------------------------------------
def generate_bot_reply(user_message: str) -> str:
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system",
                 "content": "너는 어린이를 상냥하게 도와주는 귀여운 상담 챗봇 '봉봉'이다."},
                {"role": "user", "content": user_message}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"(오류 발생) {str(e)}"


# -------------------------------------------------
# 세션 초기화
# -------------------------------------------------
def init_session():
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    if "first_loaded" not in st.session_state:
        st.session_state["first_loaded"] = False


# -------------------------------------------------
# 메시지 저장
# -------------------------------------------------
def add_message(role: str, text: str):
    st.session_state["messages"].append({
        "role": role,              # user 또는 bot
        "message": text,
        "timestamp": datetime.now().isoformat()
    })


# -------------------------------------------------
# 메시지 렌더링 (말풍선 UI)
# -------------------------------------------------
def render_chat_messages():
    for msg in st.session_state["messages"]:
        if msg["role"] == "bot":
            # 왼쪽 말풍선
            st.markdown(
                f"""
                <div style="text-align:left;">
                    <div style="
                        display:inline-block;
                        background:#f1f0f0;
                        padding:12px 15px;
                        border-radius:12px;
                        margin:5px 0;
                        max-width:70%;
                        font-size:16px;">
                        🤖 <b>봉봉</b><br>{msg['message']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            # 오른쪽 말풍선
            st.markdown(
                f"""
                <div style="text-align:right;">
                    <div style="
                        display:inline-block;
                        background:#d1e7ff;
                        padding:12px 15px;
                        border-radius:12px;
                        margin:5px 0;
                        max-width:70%;
                        font-size:16px;">
                        🌟 <b>나</b><br>{msg['message']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# -------------------------------------------------
# 첫 인사 (룰베이스)
# -------------------------------------------------
def send_first_greeting():
    greeting = (
        "안녕! 나는 너의 마음을 함께 들여다봐주는 친구 ‘봉봉’이야 😊\n\n"
        "오늘 어떤 마음으로 왔어?"
    )
    add_message("bot", greeting)
    
    
# -------------------------------------------------
# CSV 저장 함수 (모든 row에 세션 정보 포함)
# -------------------------------------------------
def save_as_csv():
    msgs = st.session_state["messages"]

    # 세션 메타데이터
    session_id = "sess_001"
    user_id = "user_abc123"
    created_at = datetime.now().isoformat()
    chat_type = "rule_based"

    rows = []
    turn = 1

    # 메시지를 2개씩 (bot → user) turn 단위로 묶기
    for i in range(0, len(msgs), 2):
        bot_msg = msgs[i] if msgs[i]["role"] == "bot" else None
        user_msg = msgs[i+1] if i+1 < len(msgs) and msgs[i+1]["role"] == "user" else None

        # bot row
        if bot_msg:
            rows.append({
                "session_id": session_id,
                "user_id": user_id,
                "created_at": created_at,
                "chat_type": chat_type,
                "turn": turn,
                "role": "bot",
                "text": bot_msg["message"],
                "timestamp": bot_msg["timestamp"],
            })

        # user row
        if user_msg:
            rows.append({
                "session_id": session_id,
                "user_id": user_id,
                "created_at": created_at,
                "chat_type": chat_type,
                "turn": turn,
                "role": "user",
                "text": user_msg["message"],
                "timestamp": user_msg["timestamp"],
            })
        else:
            # user 발화 없을 때 빈 row
            rows.append({
                "session_id": session_id,
                "user_id": user_id,
                "created_at": created_at,
                "chat_type": chat_type,
                "turn": turn,
                "role": "user",
                "text": "",
                "timestamp": "",
            })

        turn += 1

    df = pd.DataFrame(rows)

    csv_bytes = df.to_csv(index=False).encode("utf-8-sig")

    st.download_button(
        "⬇ CSV 다운로드",
        data=csv_bytes,
        file_name="chat_turns.csv",
        mime="text/csv"
    )




# -------------------------------------------------
# JSON 저장 함수 (messages → dialogue 구조 변환)
# -------------------------------------------------
def save_as_json():
    msgs = st.session_state["messages"]

    dialogue = []
    turn_index = 1

    for i in range(0, len(msgs), 2):
        bot_msg = msgs[i] if i < len(msgs) and msgs[i]["role"] == "bot" else None
        user_msg = msgs[i+1] if (i+1) < len(msgs) and msgs[i+1]["role"] == "user" else None

        if bot_msg:
            bot_block = {
                "role": "bot",
                "text": bot_msg["message"],
                "timestamp": bot_msg["timestamp"]
            }
        else:
            bot_block = None

        if user_msg:
            user_block = {
                "role": "user",
                "text": user_msg["message"],
                "timestamp": user_msg["timestamp"]
            }
        else:
            user_block = None

        dialogue.append({
            "turn": turn_index,
            "bot": bot_block,
            "user": user_block
        })

        turn_index += 1

    data = {
        "session_id": "sess_001",
        "user_id": "user_abc123",
        "created_at": datetime.now().isoformat(),
        "chat_type": "rule_based",
        "dialogue": dialogue
    }

    json_str = json.dumps(data, ensure_ascii=False, indent=2)

    st.download_button(
        "⬇ JSON 다운로드",
        json_str,
        file_name="chat_history.json",
        mime="application/json"
    )



# -------------------------------------------------
# MAIN
# -------------------------------------------------
def main():
    st.title("💛 Chatbot Demo – Step 1 (GPT + 말풍선 UI)")

    # 세션 준비
    init_session() # 세션 초기화화

    # 첫 인사 출력
    if not st.session_state["first_loaded"]:
        send_first_greeting() # 첫 인사 출력(최초 1번)
        st.session_state["first_loaded"] = True

    # 말풍선 대화 UI 렌더링
    render_chat_messages() # 말풍선 대화 UI 렌더링(화면에 말풍선 표시시)

    # 사용자 입력창 표시
    user_input = st.chat_input("메시지를 입력해보세요!")

    if user_input:
        # 1) 사용자 메시지 먼저 화면에 표시
        add_message("user", user_input)

        # 2) rerun → 사용자 말풍선 먼저 표시
        st.rerun() #mac

    # GPT 응답은 화면 갱신 후에 처리
    if len(st.session_state["messages"]) > 0:
        last_msg = st.session_state["messages"][-1]
        if last_msg["role"] == "user" and not last_msg.get("responded", False):

            # GPT 응답 생성
            bot_reply = generate_bot_reply(last_msg["message"])
            add_message("bot", bot_reply)

            # 중복 응답 방지
            last_msg["responded"] = True

            # 화면 갱신 (봇 말풍선 표시)
            st.rerun()

    # 저장 영역
    st.markdown("---")
    st.subheader("📥 대화 저장")

    save_as_json()
    save_as_csv()


# 실행
if __name__ == "__main__":
    main()

