import os
import sys
import json
from datetime import datetime
import streamlit as st
from dotenv import load_dotenv
import pandas as pd
import streamlit.components.v1 as components  # 대화 저장 후 스크롤업 방지용

# 환경 변수 로드
load_dotenv()

# 프로젝트 루트 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

# OpenAI (데모2에서는 사용 안 하지만, 구조 유지 차원에서 그대로 둠)
from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -------------------------------------------------
# (참고) GPT-4o 챗봇 응답 함수 - 데모2에서는 사용 X
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
# 5개 고정 질문 리스트 (데모2 전용)
# -------------------------------------------------
QUESTIONS = [
    "그림 속 너는 지금 무엇을 하고 있어?",
    "그림 속 너는 지금 어떤 기분이야?",
    "오늘은 왜 이렇게 그리고 싶었어?",
    "그림 속 너에게 해주고 싶은 말은 뭐야?",
    "내일의 너는 어떤 모습이면 좋겠어?"
]


# -------------------------------------------------
# 세션 초기화
# -------------------------------------------------
def init_session():
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    if "first_loaded" not in st.session_state:
        st.session_state["first_loaded"] = False

    # 데모2용 상태값
    if "question_index" not in st.session_state:
        st.session_state["question_index"] = 0  # 몇 번째 질문까지 보냈는지

    if "all_answered" not in st.session_state:
        st.session_state["all_answered"] = False  # 5문항 모두 답했는지

    if "downloads_enabled" not in st.session_state:
        st.session_state["downloads_enabled"] = False  # JSON/CSV 다운로드 활성 여부


# -------------------------------------------------
# 턴 단위 파일 저장 (실시간 append)
# -------------------------------------------------
def append_turn_to_file(role, text):
    # messages 배열의 마지막 index 기준으로 턴 번호 계산
    current_index = len(st.session_state["messages"]) - 1
    turn_number = (current_index // 2) + 1  # bot+user = 1턴 단위

    log = {
        "session_id": "sess_001",
        "timestamp": datetime.now().isoformat(),
        "role": role,
        "text": text,
        "turn": turn_number
    }

    # -----------------------------------------
    # 저장 경로(data/logs/chat_log.jsonl) 설정
    # -----------------------------------------
    log_dir = os.path.join(os.path.dirname(__file__), "../../data/logs")
    os.makedirs(log_dir, exist_ok=True)

    log_path = os.path.join(log_dir, "chat_log.jsonl")

    # JSONL 형식 저장
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(log, ensure_ascii=False) + "\n")


# -------------------------------------------------
# 메시지 저장 (세션 + 파일)
# -------------------------------------------------
def add_message(role: str, text: str):
    st.session_state["messages"].append({
        "role": role,              # user 또는 bot
        "message": text,
        "timestamp": datetime.now().isoformat()
    })

    # 턴 단위 파일 실시간 저장
    append_turn_to_file(role, text)


# -------------------------------------------------
# 다음 질문 보내기 (중복 없이 순서대로)
# -------------------------------------------------
def send_next_question_if_needed():
    q_idx = st.session_state["question_index"]

    # 이미 모든 질문을 다 보냈다면 종료
    if q_idx >= len(QUESTIONS):
        return

    msgs = st.session_state["messages"]

    # 첫 로딩이거나, 직전에 유저가 답변을 한 경우에만 다음 질문 발화
    if len(msgs) == 0 or msgs[-1]["role"] == "user":
        question_text = QUESTIONS[q_idx]
        add_message("bot", question_text)
        st.session_state["question_index"] += 1


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
                        font-size:16px;
                        color:#000000;">
                        🧸 <b>봉봉</b><br>{msg['message']}
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
                        font-size:16px;
                        color:#000000;">
                        🌟 <b>나</b><br>{msg['message']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# -------------------------------------------------
# 첫 인사 (데모2에서는 사용 X, 구조 유지를 위해 남겨둠)
# -------------------------------------------------
def send_first_greeting():
    greeting = (
        "안녕! 나는 너의 마음을 함께 들여다봐주는 친구 ‘봉봉’이야 😊\n\n"
        "오늘 어떤 마음으로 왔어?"
    )
    add_message("bot", greeting)


# -------------------------------------------------
# CSV 저장 함수 (모든 row에 세션 정보 포함)
#   disabled=True 이면 버튼 비활성화
# -------------------------------------------------
def save_as_csv(disabled: bool = False):
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
        mime="text/csv",
        disabled=disabled
    )


# -------------------------------------------------
# JSON 저장 함수 (messages → dialogue 구조 변환)
#   disabled=True 이면 버튼 비활성화
# -------------------------------------------------
def save_as_json(disabled: bool = False):
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
        mime="application/json",
        disabled=disabled
    )


# -------------------------------------------------
# MAIN
# -------------------------------------------------
def main():

    # 오른쪽 상단 담당자 표시
    st.markdown(
        """
        <style>
        .top-right-info {
            position: absolute;
            top: 10px;
            right: 20px;
            font-size: 14px;
            color: #999;
        }
        </style>
        <div class="top-right-info">
            (담당자: 미술인지심리연구소 심기섭)
        </div>
        """,
        unsafe_allow_html=True
    )

    # 타이틀 표시(중앙정렬) - 데모2
    st.markdown(
        """
        <div style='text-align:center; margin-top: 20px; margin-bottom: 30px;'>
            <div style='font-size: 34px; font-weight: 700;'>
                💛 Chatbot Demo – Step 2
            </div>
            <div style='font-size: 26px; font-weight: 500; margin-top: -5px;'>
                (5문항 고정 질문 챗봇)
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 세션 준비
    init_session()

    # 🔹 최초 1회만 첫 질문 출력
    if not st.session_state["first_loaded"]:
        send_next_question_if_needed()   # 질문 1번 출력
        st.session_state["first_loaded"] = True
        st.rerun()   # 🔥 첫 질문도 파일에 저장되도록 강제 재실행

    # ----------------------------
    # 사용자 입력 처리 (5문항 완료 전까지만)
    # ----------------------------
    user_input = None
    if not st.session_state["all_answered"]:
        user_input = st.chat_input("봉봉에게 네 마음을 이야기해줘 😊")

    if user_input and not st.session_state["all_answered"]:
        # 1) 사용자 답변 저장
        add_message("user", user_input)

        # 2) 아직 질문이 남아 있으면 다음 질문 바로 추가
        if st.session_state["question_index"] < len(QUESTIONS):
            send_next_question_if_needed()

        # 3) 질문·답변 개수 모두 충족했는지 체크
        if st.session_state["question_index"] >= len(QUESTIONS):
            user_msgs = [m for m in st.session_state["messages"] if m["role"] == "user"]
            if len(user_msgs) >= len(QUESTIONS):
                st.session_state["all_answered"] = True

        # 4) 아직 다 안 끝났으면 화면 다시 그리기
        if not st.session_state["all_answered"]:
            st.rerun()

    # ----------------------------
    # (입력 처리 후) 지금까지의 대화 UI 렌더링
    # ----------------------------
    render_chat_messages()

    # ----------------------------
    # 5문항 모두 완료된 경우: 입력창 없이 "대화 저장" 버튼만 중앙에 표시
    # ----------------------------
    if st.session_state["all_answered"]:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            save_clicked = st.button("대화 저장", use_container_width=True)

        if save_clicked:
            st.session_state["downloads_enabled"] = True

    # ----------------------------
    # 저장 영역 (JSON / CSV 다운로드)
    # ----------------------------
    st.markdown("---")
    st.subheader("📥 대화 저장")

    downloads_enabled = st.session_state.get("downloads_enabled", False)

    # all_answered + 대화 저장 버튼 클릭 전에는 비활성화
    save_as_json(disabled=not downloads_enabled)
    save_as_csv(disabled=not downloads_enabled)


    # --------------------------------------------------------------------------------
    # [최종 수정] 조건부 실행
    # 평소 대화 중(채팅)일 때는 Streamlit 기본 스크롤을 따르고(렉 방지),
    # '대화 저장' 버튼을 눌러서 화면이 길어지는 순간에만! 강제로 끌어내립니다.
    # --------------------------------------------------------------------------------
    
    # 만약 '대화 저장' 버튼을 눌러서 다운로드 창이 열린 상태라면? -> 강력한 스크롤 가동
    if st.session_state.get("downloads_enabled"):
        # 1. 화면 맨 아래에 포커스용 자석 태그 생성
        st.markdown(
            """
            <div id="bottom-magnet" tabindex="0" style="height: 1px; width: 100%; visibility: hidden;"></div>
            """,
            unsafe_allow_html=True
        )

        # 2. 자바스크립트로 강제 포커스 (이때만 실행됨)
        # behavior: 'auto'를 명시하여 스크롤을 즉각적으로 이동시킵니다. (사람 인식 최소화)
        js_code = """
        <script>
            function anchorAtBottom() {
                var magnet = window.parent.document.getElementById("bottom-magnet");
                if (magnet) {
                    magnet.scrollIntoView({
                        block: "end", 
                        inline: "nearest", 
                        behavior: "auto" // <--- 핵심 수정: 즉각적인 스크롤 이동
                    });
                    // magnet.focus(); // focus()는 옵션이며, 즉각 이동에는 필수 아님.
                }
            }
            // Streamlit 렌더링 후 DOM이 로드될 시간에 맞춰 빠르게 실행
            setTimeout(anchorAtBottom, 50); // 시간을 더 단축 (50ms)
            setTimeout(anchorAtBottom, 150);
            setTimeout(anchorAtBottom, 250);
        </script>
        """
        # components.html의 height는 0을 유지하여 공간을 차지하지 않도록 합니다.
        components.html(js_code, height=0)



# 실행
if __name__ == "__main__":
    main()
