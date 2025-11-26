import os
import sys
import json
from datetime import datetime
import streamlit as st
from dotenv import load_dotenv
import pandas as pd
from openai import OpenAI

# Streamlit 스크롤 방지용 컴포넌트
import streamlit.components.v1 as components 

# 환경 변수 로드
load_dotenv()

# OpenAI 클라이언트
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))



# ------------------------------
# Memory Loader
# ------------------------------
def load_static_memory():
    base = os.path.dirname(__file__)
    path = os.path.join(base, "data/memory/static_memory.json")

    # 폴더만 생성 (파일은 생성하지 않음!)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # 파일이 이미 만들어져 있다면 → 그대로 읽기
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    # 파일이 원래 없었던 경우 → 기본 빈 메모리 반환
    return {"static_memory": {}}



def load_dynamic_memory():
    base = os.path.dirname(__file__)
    path = os.path.join(base, "data/memory/dynamic_memory.json")

    # 폴더가 없으면 폴더도 생성
    dir_path = os.path.dirname(path)
    os.makedirs(dir_path, exist_ok=True)

    # 파일 없으면 기본 구조 반환
    if not os.path.exists(path):
        return {"dynamic_memory": {"turns": []}}

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)



def save_dynamic_memory(dynamic):
    base = os.path.dirname(__file__)
    path = os.path.join(base, "data/memory/dynamic_memory.json")
    dir_path = os.path.dirname(path)
    os.makedirs(dir_path, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump({"dynamic_memory": dynamic}, f, ensure_ascii=False, indent=2)



def get_memory_context():
    static = load_static_memory().get("static_memory", {})
    dynamic = load_dynamic_memory().get("dynamic_memory", {})
    return static, dynamic


# 동적 메모리 업데이트 함수
def update_dynamic_memory(role, text):
    dynamic = load_dynamic_memory()["dynamic_memory"]

    dynamic["turns"].append({
        "role": role,
        "text": text
    })

    save_dynamic_memory(dynamic)



# ------------------------------
# 고정 질문
# ------------------------------
RULE_QUESTIONS = {
    1: "친구야, 오늘 어땠어?",
    2: "오늘 활동 중에 가장 기억에 남았던 순간은 뭐였어?",
    3: "마지막으로, 오늘 활동을 마치며 봉봉이에게 하고 싶은 말 있을까?"
}

# ------------------------------
# GPT FUNCTIONS
# ------------------------------

def build_memory_prompt(static, dynamic):
    # 정적 메모리
    s = static
    d = dynamic.get("turns", [])

    static_block = f"""
[학생 정보 요약 — 참고용 메모리]
- 자기표현 키워드: {s.get('user_self_keywords')}
- 그림 제목: {s.get('user_drawing_info', {}).get('title')}
- 그림 속 나이: {s.get('user_drawing_info', {}).get('age_in_picture')}
- 현재 행동: {s.get('user_drawing_info', {}).get('current_action')}
- 미래 예측: {s.get('user_drawing_info', {}).get('future_prediction')}
- 그림 속 메시지: {s.get('user_drawing_info', {}).get('message_to_self')}

[강점 및 성향]
- 좋아하는 것: {s.get('user_hero_info', {}).get('likes')}
- 잘하는 것: {s.get('user_hero_info', {}).get('abilities')}
- 강점: {s.get('user_hero_info', {}).get('strength_points')}
- 약점: {s.get('user_hero_info', {}).get('weakness_points')}
- 잠재력: {s.get('user_hero_info', {}).get('potentials')}
"""

    # 동적 메모리 = 지금까지의 대화 축약
    turns_text = "\n".join(
        [f"- {t['role']}: {t['text']}" for t in d[-10:]]  # 최근 10개만 사용
    )

    dynamic_block = f"""
[지금까지의 대화 내용(최근)]
{turns_text}
"""

    return static_block + "\n" + dynamic_block




def gpt_free_followup(user_message: str, stage: int, turn: int) -> str:
     # 정적·동적 메모리 불러오기
    static, dynamic = get_memory_context()

    # memory prompt 생성
    memory_text = build_memory_prompt(static, dynamic)
    
    print(static)
    print(dynamic)
    print(memory_text)
    print("--------------------------------")
    """공감 + 자유맥락 후속 질문 (고정질문 X)"""
    stage_label = {1: "S1 활동묻기 단계", 2: "S2 기억회상 단계", 3: "S3 활동 마무리 단계"}.get(stage, "대화 단계")

    prompt = f"""
{memory_text}

[상황]
- 지금은 {stage_label}에서 공감 대화를 이어가고 있어.
- 지금은 공감 {turn}번째 턴이야.
- 이 턴에서는 고정 질문 대신, 아이 말에 기반한 자유로운 질문을 사용해.
- 현재는 자기 자신에 대한 그림을 그리고 활동을 마무리 하고 있는 단계에서 봉봉이 너가 말하는 상황이야.(과거, 미래시점으로 이야기 하지 않기, 해당 활동을 너도 함께 이해하고 있는 상황 전제를 하고 대답할 것)

[해야 할 일]
1) 아래 아이의 말에 진심 어린 공감/격려를 2~3문장으로 작성.
2) 이어서 방금 아이가 한 말에서 자연스럽게 이어지는 자유 질문 1개 제시.
3) 전체를 하나의 발화처럼 자연스럽게 이어서 출력.

[중요 규칙]
- 장난/의미 없는 입력은 감정 신호로만 이해하고 부드럽게 정돈해서 반응하기.
- 분석/평가/단정/지적 금지.
- 단계나 시스템 용어 언급 금지.

[아이의 말]
{user_message}
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            # 1) 역할 지시 — 여기에서만
            {
                "role": "system",
                "content": (
                    "너는 어린이를 따뜻하게 도와주는 상담 챗봇 '봉봉'이다. "
                    "친근한 반말로 말하고, 아이가 장난을 쳐도 부드럽게 정돈해 반응한다. "
                    "아래 규칙을 절대 어기지 말 것:\n"
                    "- 단계명(S1/S2/S3 등) 언급 금지\n"
                    "- 분석/평가/솔루션/지적 금지\n"
                    "- 반복된 질문 금지 (특히 직전 턴에서 이미 물어본 질문)\n"
                    "- 고정 질문을 임의로 생성하거나 반복 금지\n"
                    "- 마지막 closing에서는 질문 금지\n"
                )
            },

            # 2) 유저 메시지 — 전체 prompt 전달
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content


def gpt_intro_with_fixed(prev_answer: str, stage: int, fixed_question: str) -> str:
    static, dynamic = get_memory_context()
    memory_text = build_memory_prompt(static, dynamic)
    print(static)
    print(dynamic)
    print(memory_text)
    print("--------------------------------")
    """단계 시작 시: 직전 답변 공감 + 고정 질문"""
    stage_label = {2: "S2 기억회상 단계", 3: "S3 활동 마무리 단계"}.get(stage, "다음 단계")

    prompt = f"""

{memory_text}

[상황]
- 지금은 {stage_label}의 첫 턴이야.
- 현재는 자기 자신에 대한 그림을 그리고 활동을 마무리 하고 있는 단계에서 봉봉이 너가 말하는 상황이야.(과거, 미래시점으로 이야기 하지 않기, 해당 활동을 너도 함께 이해하고 있는 상황 전제를 하고 대답할 것)
- 직전 단계 마지막에 아이가 이렇게 말했어:
"{prev_answer}"

[해야 할 일]
1) 아이의 말을 2~3문장으로 진심 어린 공감/격려.
2) 이어서 아래 고정 질문 문장을 딱 한 번 자연스럽게 포함해 묻기.



[중요 규칙]
- 장난/의미 없는 입력은 감정 신호로만 이해하고 부드럽게 정돈해서 반응하기.
- 분석/평가/단정/지적 금지.
- 단계나 시스템 용어 언급 금지.

[고정 질문]
"{fixed_question}"
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            # 1) 역할 지시 — 여기에서만
            {
                "role": "system",
                "content": (
                    "너는 어린이를 따뜻하게 도와주는 상담 챗봇 '봉봉'이다. "
                    "친근한 반말로 말하고, 아이가 장난을 쳐도 부드럽게 정돈해 반응한다. "
                    "아래 규칙을 절대 어기지 말 것:\n"
                    "- 단계명(S1/S2/S3 등) 언급 금지\n"
                    "- 분석/평가/솔루션/지적 금지\n"
                    "- 반복된 질문 금지 (특히 직전 턴에서 이미 물어본 질문)\n"
                    "- 고정 질문을 임의로 생성하거나 반복 금지\n"
                    "- 마지막 closing에서는 질문 금지\n"
                )
            },

            # 2) 유저 메시지 — 전체 prompt 전달
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content


def gpt_closing(user_message: str) -> str:
    static, dynamic = get_memory_context()
    memory_text = build_memory_prompt(static, dynamic)

    prompt = f"""
{memory_text}

[상황]
- 지금은 오늘 활동을 마무리하는 마지막 턴이야.
- 현재는 자기 자신에 대한 그림을 그리고 활동을 마무리 하고 있는 단계에서 봉봉이 너가 말하는 상황이야.(과거, 미래시점으로 이야기 하지 않기, 해당 활동을 너도 함께 이해하고 있는 상황 전제를 하고 대답할 것)

[해야 할 일]
1) 아이의 마지막 말을 바탕으로 2~3문장 공감·정리·격려.
2) 마지막 1문장은 절대 질문으로 끝나면 안되며, 감사 또는 오늘 느낀 점을 가볍게 다시 떠올리게 하는 마무리 문장이면서도 마지막에 "안녕"이라고 인사를 꼭 해줘.
3) 마지막 1문장 주의사항: 질문으로 끝나면 안됨, 너무 장황하게 길게 말하지 않기, 나중에 또 만날것 처럼 마무리 멘트 하지 않기(예: 궁금한게 있으면 언제든지 물어봐!)

[중요 규칙]
- 장난/의미 없는 입력은 감정 신호로만 이해하고 부드럽게 정돈해서 반응하기.
- 아이의 답변에 분석/평가/단정/지적 금지.
- 단계나 시스템 용어 언급 금지.

[아이의 마지막 말]
{user_message}
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            # 1) 역할 지시 — 여기에서만
            {
                "role": "system",
                "content": (
                    "너는 지금 마지막 턴에서 마무리 인사를 해야 한다. "
                    "절대 질문을 하지 말고, 문장을 물음표로 끝내지 마라. "
                    "미래 유도 멘트(예: '또 보자', '언제든지 물어봐')를 사용하지 말고, "
                    "마지막 문장은 반드시 '안녕'으로 끝내라."
                )
            },

            # 2) 유저 메시지 — 전체 prompt 전달
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content



# -------------------------------------------------
# 메시지 추가 (로그 저장 통합)
# -------------------------------------------------
def add_message(role: str, text: str):
    st.session_state["messages"].append({
        "role": role,
        "message": text,
        "timestamp": datetime.now().isoformat()
    })
    
    # 턴 단위 파일 실시간 저장
    append_turn_to_file(role, text)

     # 동적 메모리 업데이트 추가
    update_dynamic_memory(role, text)


# -------------------------------------------------
# 세션 초기화
# -------------------------------------------------
def init_session():
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    if "state" not in st.session_state:
        st.session_state["state"] = 1  # 1=S1, 2=S2, 3=S3

    if "substep" not in st.session_state:
        st.session_state["substep"] = 1  # 1~6

    if "downloads_enabled" not in st.session_state:
        st.session_state["downloads_enabled"] = False


# -------------------------------------------------
# 턴 단위 파일 저장 (실시간 append) - 데모 1에서 가져옴
# -------------------------------------------------
def append_turn_to_file(role, text):
    current_index = len(st.session_state["messages"]) - 1
    turn_number = (current_index // 2) + 1
    
    log = {
        "session_id": "sess_004", # 세션 ID는 데모 4를 반영하여 변경
        "timestamp": datetime.now().isoformat(),
        "role": role,
        "text": text,
        "turn": turn_number
    }
    
    # 저장 경로 설정
    log_dir = os.path.join(os.path.dirname(__file__), "data/logs") # 실제 환경에 맞게 경로 조정
    os.makedirs(log_dir, exist_ok=True)

    log_path = os.path.join(log_dir, "chat_log.jsonl")

    # JSONL 형식 저장
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(log, ensure_ascii=False) + "\n")




# -------------------------------------------------
# UI 렌더링
# -------------------------------------------------
def render_chat_messages():
    # 빈 공간을 만들어 스크롤을 맨 아래로 내리는 역할을 합니다.
    # st.empty().markdown(...) 대신, div를 직접 사용하여 Streamlit 기본 동작을 보장합니다.
    for msg in st.session_state["messages"]:
        if msg["role"] == "bot":
            st.markdown(f"""
            <div style="text-align:left;">
                <div style="
                    display:inline-block; background:#f1f0f0;
                    padding:12px 15px; border-radius:12px;
                    margin:5px 0; max-width:70%;
                    font-size:16px;
                    color:#000000;">
                    🧸 <b>봉봉</b><br>{msg['message']}
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="text-align:right;">
                <div style="
                    display:inline-block; background:#d1e7ff;
                    padding:12px 15px; border-radius:12px;
                    margin:5px 0; max-width:70%;
                    font-size:16px;
                    color:#000000;">
                    🌟 <b>나</b><br>{msg['message']}
                </div>
            </div>
            """, unsafe_allow_html=True)


# -------------------------------------------------
# CSV 저장 함수 (데모 1에서 가져옴)
# -------------------------------------------------
def save_as_csv(disabled: bool = False):
    msgs = st.session_state["messages"]
    if not msgs:
        st.download_button("⬇ CSV 다운로드", data="", file_name="chat_turns.csv", mime="text/csv", disabled=True)
        return

    session_id = "sess_004"
    user_id = "user_abc123"
    created_at = datetime.now().isoformat()
    chat_type = "fsm_empathy_2turn"

    rows = []
    turn = 1

    for i in range(0, len(msgs), 2):
        # 봇과 유저 메시지를 묶어 턴을 만듭니다.
        bot_msg = msgs[i] if msgs[i]["role"] == "bot" else None
        user_msg = msgs[i+1] if i+1 < len(msgs) and msgs[i+1]["role"] == "user" else None

        # bot row
        if bot_msg:
            rows.append({
                "session_id": session_id, "user_id": user_id, "created_at": created_at,
                "chat_type": chat_type, "turn": turn, "role": "bot",
                "text": bot_msg["message"], "timestamp": bot_msg["timestamp"],
            })

        # user row
        if user_msg:
            rows.append({
                "session_id": session_id, "user_id": user_id, "created_at": created_at,
                "chat_type": chat_type, "turn": turn, "role": "user",
                "text": user_msg["message"], "timestamp": user_msg["timestamp"],
            })
        elif bot_msg:
            # 봇 발화만 있고 유저 발화가 없는 경우 빈 row 추가 (마지막 턴 등)
             rows.append({
                "session_id": session_id, "user_id": user_id, "created_at": created_at,
                "chat_type": chat_type, "turn": turn, "role": "user",
                "text": "", "timestamp": "",
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
# JSON 저장 함수 (데모 1에서 가져옴)
# -------------------------------------------------
def save_as_json(disabled: bool = False):
    msgs = st.session_state["messages"]
    if not msgs:
        st.download_button("⬇ JSON 다운로드", data="", file_name="chat_history.json", mime="application/json", disabled=True)
        return

    dialogue = []
    turn_index = 1

    for i in range(0, len(msgs), 2):
        bot_msg = msgs[i] if i < len(msgs) and msgs[i]["role"] == "bot" else None
        user_msg = msgs[i+1] if (i+1) < len(msgs) and msgs[i+1]["role"] == "user" else None

        bot_block = {"role": "bot", "text": bot_msg["message"], "timestamp": bot_msg["timestamp"]} if bot_msg else None
        user_block = {"role": "user", "text": user_msg["message"], "timestamp": user_msg["timestamp"]} if user_msg else None

        dialogue.append({
            "turn": turn_index,
            "bot": bot_block,
            "user": user_block
        })
        turn_index += 1

    data = {
        "session_id": "sess_004",
        "user_id": "user_abc123",
        "created_at": datetime.now().isoformat(),
        "chat_type": "fsm_empathy_2turn",
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
# Main FSM Process (안정화된 최종 구조)
# -------------------------------------------------

def process_flow(user_input=None):
    """
    process_flow(user_input)

    챗봇의 전체 대화 단계를 관리하는 Finite State Machine(FSM).

    🔹 FSM 설계
    S1 → S2 → S3 순서로 진행되며,
    각 스테이지에서 GPT 공감 턴은 고정된 substep(1~6)으로 나뉜다.

    - substep 1 : RULE 또는 GPT의 첫 질문 자동 발화
    - substep 2 : 사용자 입력
    - substep 3 : GPT 공감 1턴 자동 발화
    - substep 4 : 사용자 입력
    - substep 5 : GPT 공감 2턴 자동 발화
    - substep 6 : 사용자 입력 후 다음 스테이지로 전환 (S3에서는 종료)

    🔹 종료 상태 (End State)
    state==3 AND substep==6 → 대화 종료
    - 입력창 숨김
    - 자동 발화 없음
    - 다음 스테이지 이동 없음
    - 다운로드 버튼만 노출
    """
    
    # ---------------------------------------------
    # 🔥 0. 대화 종료 후 입력/자동진행 완전 차단
    # ---------------------------------------------
    # state=3, substep=6 = S3 마지막 GPT 발화까지 모두 끝난 상태
    if st.session_state["state"] == 3 and st.session_state["substep"] == 6:
        return  # 더 이상 어떤 처리도 하지 않음

    state = st.session_state["state"]
    sub = st.session_state["substep"]

    # GPT 답변 중(sub=1,3,5)에 들어온 유저 입력은 무시
    if user_input and sub not in [2, 4, 6]:
        return

    # -------------------------------------------------
    # 1. 유저 입력 처리 (sub 2, 4, 6)
    # -------------------------------------------------
    if user_input:
        add_message("user", user_input)

        # substep 2, 4 → 다음 GPT 자동 발화를 호출해야 함
        if sub in [2, 4]:
            st.session_state["substep"] += 1
            st.rerun()

        # substep 6 → 다음 단계로 넘어감
        elif sub == 6:
            if state < 3:
                st.session_state["state"] += 1
                st.session_state["substep"] = 1
                st.rerun()
            else:
                # state=3 & sub=6은 위 최상단 차단에서 이미 필터됨
                return

        return  # user_input 처리 종료

    

    # -------------------------------------------------
    # 2. GPT/RULE 자동 발화 처리 (user_input == None일 때)
    # 이 부분이 첫 로딩 시점(sub=1)과 GPT 응답 턴(sub=3, 5)을 담당
    # -------------------------------------------------
    
    # S1 활동묻기
    if state == 1:
        if sub == 1: # S1-1: 룰베이스 고정 질문 (첫 로딩 시점)
            add_message("bot", RULE_QUESTIONS[1])
            st.session_state["substep"] = 2
            st.rerun() # 유저 입력 대기 상태로 전환
        
        if sub == 3: # S1-3: GPT 공감 1턴 (직전 유저 답변 후)
            last = st.session_state["messages"][-1]["message"]
            bot_msg = gpt_free_followup(last, 1, 1)
            add_message("bot", bot_msg)
            st.session_state["substep"] = 4
            st.rerun() # 유저 입력 대기 상태로 전환
        
        if sub == 5: # S1-5: GPT 공감 2턴
            last = st.session_state["messages"][-1]["message"]
            bot_msg = gpt_free_followup(last, 1, 2)
            add_message("bot", bot_msg)
            st.session_state["substep"] = 6
            st.rerun() # 유저 입력 대기 상태로 전환

    # S2 기억회상
    elif state == 2:
        if sub == 1: # S2-1: GPT 공감 + 고정 질문 (S1 종료 후)
            prev_answer = st.session_state["messages"][-1]["message"]
            fixed = RULE_QUESTIONS[2]
            bot_msg = gpt_intro_with_fixed(prev_answer, 2, fixed)
            add_message("bot", bot_msg)
            st.session_state["substep"] = 2
            st.rerun()
            
        if sub == 3: # S2-3: GPT 공감 1턴
            last = st.session_state["messages"][-1]["message"]
            bot_msg = gpt_free_followup(last, 2, 1)
            add_message("bot", bot_msg)
            st.session_state["substep"] = 4
            st.rerun()
            
        if sub == 5: # S2-5: GPT 공감 2턴
            last = st.session_state["messages"][-1]["message"]
            bot_msg = gpt_free_followup(last, 2, 2)
            add_message("bot", bot_msg)
            st.session_state["substep"] = 6
            st.rerun()

    # S3 마무리
    elif state == 3:
        if sub == 1: # S3-1: GPT 공감 + 고정 질문 (S2 종료 후)
            prev_answer = st.session_state["messages"][-1]["message"]
            fixed = RULE_QUESTIONS[3]
            bot_msg = gpt_intro_with_fixed(prev_answer, 3, fixed)
            add_message("bot", bot_msg)
            st.session_state["substep"] = 2
            st.rerun()
            
        if sub == 3: # S3-3: GPT 공감 1턴
            last = st.session_state["messages"][-1]["message"]
            bot_msg = gpt_free_followup(last, 3, 1)
            add_message("bot", bot_msg)
            st.session_state["substep"] = 4
            st.rerun()
            
        if sub == 5: # S3-5: GPT 공감 2턴 (마무리 발화)
            last = st.session_state["messages"][-1]["message"]
            bot_msg = gpt_closing(last)
            add_message("bot", bot_msg)
            st.session_state["substep"] = 6
            #st.session_state["downloads_enabled"] = True
            st.rerun() # 대화 완료 후 다운로드 버튼을 활성화하기 위해 RERUN


# -------------------------------------------------
# MAIN
# -------------------------------------------------
def main():
    st.set_page_config(layout="centered", page_title="Chatbot Demo – Step 4")

    st.markdown("""
        <style>
        .top-right-info {
            position:absolute; top:10px; right:20px;
            font-size:14px; color:#999;
        }
        div[role="radiogroup"] > label { margin-bottom: 5px; }
        .stSuccess { text-align: center; }
        </style>
        <div class="top-right-info">
            (담당자: 미술인지심리연구소 심기섭)
        </div>
    """, unsafe_allow_html=True)

    # 타이틀
    st.markdown("""
        <div style='text-align:center; margin-top: 20px; margin-bottom: 30px;'>
            <div style='font-size: 34px; font-weight: 700;'>💛 Chatbot Demo – Step 4</div>
            <div style='font-size: 24px; font-weight: 500; margin-top: -5px;'>(S1 → S2 → S3 공감 2턴 챗봇)</div>
        </div>
    """, unsafe_allow_html=True)

    init_session()

    # 1. 렌더링 (이전 세션 상태)
    render_chat_messages()

    ## 2. 사용자 입력 감지
    # user_input = st.chat_input("봉봉에게 마음을 이야기해줘 😊")


    state = st.session_state["state"]
    sub = st.session_state["substep"]
    downloads_enabled = st.session_state.get("downloads_enabled", False)

    # # S3 마지막(sub=6) + 아직 다운로드 버튼 안 누른 상태 → 입력창 숨김
    # if state == 3 and sub == 6 and not downloads_enabled:
    #     user_input = None
    # else:
    #     user_input = st.chat_input("봉봉에게 마음을 이야기해줘 😊")

    # # S3 마지막(sub=6) → 입력창은 항상 숨김
    # if state == 3 and sub == 6:
    #     user_input = None
    # else:
    #     user_input = st.chat_input("봉봉에게 마음을 이야기해줘 😊")



    # --- 입력 가능 substep 정의: 유저 입력 턴만 가능 ---
    can_user_input = (sub in [2, 4, 6])

    # --- S3 종료(sub=6)면 입력창 숨김 ---
    if state == 3 and sub == 6:
        can_user_input = False

    # --- 입력창 표시 ---
    if can_user_input:
        user_input = st.chat_input("봉봉에게 마음을 이야기해줘 😊")
    else:
        user_input = None



    # 3. FSM 처리 (입력 유무에 따라 한 번만 호출)
    process_flow(user_input)
    

    # 4. 저장 영역 (S3-5 → substep=6 이후 처리)
    state = st.session_state["state"]
    sub = st.session_state["substep"]
    downloads_enabled = st.session_state.get("downloads_enabled", False)

    # (1) 아직 다운로드 버튼 누르기 전 → 중앙에 "대화 저장" 버튼만 표시
    if state == 3 and sub == 6 and not downloads_enabled:

        st.markdown("<div style='height:25px;'></div>", unsafe_allow_html=True)  # 마무리 완료 버튼 위 여백 추가

        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            save_click = st.button("마무리 완료", use_container_width=True)

        if save_click:
            st.session_state["downloads_enabled"] = True
            st.rerun()

        # # 구분선 + 여백
        # 👉 여기! 줄 1개 생성 (딱 이거만 남겨둠)
        st.markdown(
            "<div style='margin-top:35px; margin-bottom:20px; border-top:1px solid #666;'></div>",
            unsafe_allow_html=True
        )

        

    # (2) 다운로드 버튼을 눌렀을 때 → JSON/CSV 표시
    # elif downloads_enabled:
    #     st.markdown("---")
    #     st.subheader("📥 대화 저장")

    #     col1, col2 = st.columns([1,1])
    #     with col1:
    #         save_as_json(disabled=False)
    #     with col2:
    #         save_as_csv(disabled=False)

    #     # 다운로드 후 스크롤 맨 아래 이동
    #     js_code = """
    #     <script>
    #         var body = window.parent.document.querySelector('.main');
    #         body.scrollTop = body.scrollHeight;
    #     </script>
    #     """
    #     components.html(js_code, height=0)

    elif downloads_enabled:
        st.markdown("<hr style='margin-top:35px; margin-bottom:20px;'>", unsafe_allow_html=True)
        st.subheader("📥 대화 저장")

        # JSON / CSV 버튼을 왼쪽 정렬 + 간격 좁게
        btn_area = st.columns([0.15, 0.15, 0.7])  # 왼쪽 2칸에 버튼 배치
        with btn_area[0]:
            save_as_json(disabled=False)
        with btn_area[1]:
            save_as_csv(disabled=False)

        # 스크롤 맨 아래 이동
        js_code = """
        <script>
            var body = window.parent.document.querySelector('.main');
            body.scrollTop = body.scrollHeight;
        </script>
        """
        components.html(js_code, height=0)




if __name__ == "__main__":
    main()