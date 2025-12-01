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

# 모델 환경 변수 읽기 추가
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")  # 기본값 gpt-4o

# ------------------------------
# 프롬프트 파일 경로 (외부 JSON)
# ------------------------------
PROMPTS_PATH = os.path.join(
    os.path.dirname(__file__),
    "prompts",
    "prompts.json"
)

# ------------------------------
# 고정 질문
# ------------------------------
RULE_QUESTIONS = {
    1: "친구야, 오늘 어땠어?",
    2: "오늘 활동 중에 가장 기억에 남았던 순간은 뭐였어?",
    3: "마지막으로, 오늘 활동을 마치며 봉봉이에게 하고 싶은 말 있을까?"
}

# ------------------------------
# 단계 라벨 (프롬프트에 넣는 사람용 라벨)
# ------------------------------
STAGE_LABELS = {
    1: "S1 활동묻기 단계",
    2: "S2 기억회상 단계",
    3: "S3 활동 마무리 단계",
}

# ------------------------------
# 디버그용 헬퍼
# ------------------------------
def debug_block(title: str, lines: list[str]):
    """터미널에서 보기 좋은 디버그 블록 출력."""
    print("\n" + "=" * 20 + f" {title} " + "=" * 20)
    for line in lines:
        print(line)
    print("=" * 60 + "\n")


# -------------------------------------------------
# 프롬프트 유틸 함수들
# -------------------------------------------------
def load_prompts() -> dict:
    """
    prompts/prompts.json 파일을 읽어서 dict로 반환하는 유틸 함수.
    - empathy_free_question
    - empathy_rule_question
    - empathy_ending_message
    세 가지 키를 가진 JSON 구조를 기대한다.
    """
    with open(PROMPTS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    debug_block("LOAD PROMPTS", [
        f"PROMPTS_PATH: {PROMPTS_PATH}",
        f"keys: {list(data.keys())}"
    ])
    return data


def apply_prompt_template(lines, **kwargs) -> str:
    """
    prompts.json에서 가져온 문자열 리스트(lines)를 하나의 문자열로 합치고,
    {{key}} 형태의 플레이스홀더를 kwargs로 치환한다.
    """
    text = "\n".join(lines)
    for key, value in kwargs.items():
        placeholder = "{{" + key + "}}"
        text = text.replace(placeholder, value)
    return text


def extract_question_from_reply(reply: str) -> str:
    """
    GPT 응답(reply)에서 '질문 문장'을 단순하게 추출하기 위한 보조 함수.
    - 마지막 줄부터 위로 올라가며,
      물음표(?)가 포함된 첫 번째 비어있지 않은 줄을 질문으로 본다.
    - 질문이 하나도 없으면 빈 문자열("")을 반환.
    """
    lines = reply.splitlines()
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        if "?" in line:
            return line
    return ""


def build_fixed_questions_str() -> str:
    """
    RULE_QUESTIONS 전체를 사람이 읽기 좋은 한 줄 문자열로 만들어준다.
    예: "친구야, 오늘 어땠어? / 오늘 활동 중에 가장 기억에 남았던 순간은 뭐였어? / ..."
    """
    return " / ".join(RULE_QUESTIONS[i] for i in sorted(RULE_QUESTIONS.keys()))


def build_generated_questions_str() -> str:
    """
    지금까지 생성된 자유 질문 목록을 문자열로 변환.
    - 아무것도 없으면 '현재까지 생성된 자유 질문 없음'으로 반환.
    """
    generated = st.session_state.get("generated_questions", [])
    if not generated:
        return "현재까지 생성된 자유 질문 없음"
    return " / ".join(generated)


# ------------------------------
# GPT FUNCTIONS (외부 프롬프트 사용)
# ------------------------------
def generate_empathy_free_question(user_message: str, stage: int, turn: int) -> str:
    """
    공감 + 자유 질문 생성 (고정 질문 사용 X)
    - prompts.json의 empathy_free_question 템플릿 사용
    - fixed_questions / generated_questions / stage_label / user_message를 채워서 전달
    - 생성된 응답에서 마지막 '질문 문장'을 추출해 generated_questions에 누적
    """
    # 단계 라벨 설정
    stage_label = STAGE_LABELS.get(stage, "대화 단계")

    # 프롬프트 로드 (세션 캐싱)
    prompts = st.session_state["prompts"]
    lines = prompts["empathy_free_question"]

    # 고정 질문/자유 질문 목록 문자열 생성
    fixed_questions_str = build_fixed_questions_str()
    generated_questions_str = build_generated_questions_str()

    # 템플릿 채우기
    prompt_text = apply_prompt_template(
        lines,
        stage_label=stage_label,
        user_message=user_message,
        fixed_questions=fixed_questions_str,
        generated_questions=generated_questions_str,
    )

    current_state = st.session_state.get("state")
    current_sub = st.session_state.get("substep")

    debug_block("GPT FREE QUESTION (empathy_free_question)", [
        f"[STATE] {current_state} / SUBSTEP {current_sub}",
        f"[STAGE_LABEL] {stage_label}",
        f"[TURN] {turn}",
        "",
        "[USER_MESSAGE]",
        user_message,
        "",
        "[FIXED_QUESTIONS_STR]",
        fixed_questions_str,
        "",
        "[GENERATED_QUESTIONS_STR]",
        generated_questions_str,
        "",
        "---------------- PROMPT TEXT SENT TO GPT ----------------",
        prompt_text
    ])

    # GPT 호출
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "너는 어린이를 따뜻하게 도와주는 상담 챗봇 '봉봉'이야. "
                    "친근한 반말로 말하고, 아이의 감정을 존중하면서 부드럽게 반응해."
                ),
            },
            {"role": "user", "content": prompt_text},
        ],
    )

    reply = response.choices[0].message.content

    # 생성된 응답에서 질문 문장을 추출해 자유 질문 목록에 누적
    question_line = extract_question_from_reply(reply)
    already_exists = False

    if question_line:
        if "generated_questions" not in st.session_state:
            st.session_state["generated_questions"] = []
        if question_line in st.session_state["generated_questions"]:
            already_exists = True
        st.session_state["generated_questions"].append(question_line)

    debug_block("GPT FREE QUESTION RESULT", [
        "---------------- GPT RAW RESPONSE ----------------",
        reply,
        "",
        "-------------- EXTRACTED QUESTION -----------------",
        f"EXTRACTED: {repr(question_line)}",
        f"ALREADY_EXISTS: {already_exists}",
        "",
        "----------- UPDATED GENERATED_QUESTIONS ----------",
        build_generated_questions_str()
    ])

    return reply



def generate_empathy_rule_question(prev_answer: str, stage: int, rule_question: str) -> str:
    """
    단계 시작 시: 직전 답변 공감 + (현재 턴에서 사용할) 고정 질문 1개 포함해서 묻는 함수.
    - prompts.json의 empathy_rule_question 템플릿 사용
    - fixed_questions / generated_questions / stage_label / prev_answer / rule_question 채워서 전달
    """
    stage_label = STAGE_LABELS.get(stage, "다음 단계")

    prompts = st.session_state["prompts"]
    lines = prompts["empathy_rule_question"]

    fixed_questions_str = build_fixed_questions_str()
    generated_questions_str = build_generated_questions_str()

    prompt_text = apply_prompt_template(
        lines,
        stage_label=stage_label,
        prev_answer=prev_answer,
        rule_question=rule_question,
        fixed_questions=fixed_questions_str,
        generated_questions=generated_questions_str,
    )

    current_state = st.session_state.get("state")
    current_sub = st.session_state.get("substep")

    debug_block("GPT RULE QUESTION (empathy_rule_question)", [
        f"[STATE] {current_state} / SUBSTEP {current_sub}",
        f"[STAGE_LABEL] {stage_label}",
        "",
        "[PREV_ANSWER]",
        prev_answer,
        "",
        "[RULE_QUESTION]",
        rule_question,
        "",
        "[FIXED_QUESTIONS_STR]",
        fixed_questions_str,
        "",
        "[GENERATED_QUESTIONS_STR]",
        generated_questions_str,
        "",
        "---------------- PROMPT TEXT SENT TO GPT ----------------",
        prompt_text
    ])


    # GPT 호출
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "너는 어린이를 따뜻하게 도와주는 상담 챗봇 '봉봉'이야. "
                    "친근한 반말로 말하고, 아이의 말을 먼저 공감해 준 뒤, "
                    "이번 턴에서 사용할 고정 질문을 자연스럽게 한 번만 사용해야 해."
                ),
            },
            {"role": "user", "content": prompt_text},
        ],
    )

    reply = response.choices[0].message.content

    debug_block("GPT RULE QUESTION RESULT", [
        "---------------- GPT RAW RESPONSE ----------------",
        reply
    ])

    return reply



def generate_empathy_ending_message(user_message: str) -> str:
    """
    S3 마지막 GPT 턴 — 공감 + 마무리 메시지 생성.
    - 질문 없이 끝나야 하며, 마지막 문장은 반드시 '안녕'으로 끝나야 함.
    - prompts.json의 empathy_ending_message 템플릿 사용.
    """
    prompts = st.session_state["prompts"]
    lines = prompts["empathy_ending_message"]

    fixed_questions_str = build_fixed_questions_str()
    generated_questions_str = build_generated_questions_str()

    prompt_text = apply_prompt_template(
        lines,
        user_message=user_message,
        fixed_questions=fixed_questions_str,
        generated_questions=generated_questions_str,
    )

    current_state = st.session_state.get("state")
    current_sub = st.session_state.get("substep")

    debug_block("GPT ENDING MESSAGE (empathy_ending_message)", [
        f"[STATE] {current_state} / SUBSTEP {current_sub}",
        "",
        "[USER_MESSAGE]",
        user_message,
        "",
        "[FIXED_QUESTIONS_STR]",
        fixed_questions_str,
        "",
        "[GENERATED_QUESTIONS_STR]",
        generated_questions_str,
        "",
        "---------------- PROMPT TEXT SENT TO GPT ----------------",
        prompt_text
    ])

    # GPT 호출
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "너는 오늘 활동을 마무리하는 마지막 인사를 하는 상담 챗봇 '봉봉'이야. "
                    "절대 질문을 하지 말고, 마지막 문장은 반드시 '안녕'으로 끝내야 해."
                ),
            },
            {"role": "user", "content": prompt_text},
        ],
    )


    reply = response.choices[0].message.content

    debug_block("GPT ENDING MESSAGE RESULT", [
        "---------------- GPT RAW RESPONSE ----------------",
        reply
    ])

    return reply



# -------------------------------------------------
# 세션 초기화
# -------------------------------------------------
def init_session():
    first_init = "messages" not in st.session_state

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    if "state" not in st.session_state:
        st.session_state["state"] = 1  # 1=S1, 2=S2, 3=S3

    if "substep" not in st.session_state:
        st.session_state["substep"] = 1  # 1~6

    if "downloads_enabled" not in st.session_state:
        st.session_state["downloads_enabled"] = False

    # 지금까지 생성된 자유 질문 목록 (중복 질문 방지용)
    if "generated_questions" not in st.session_state:
        st.session_state["generated_questions"] = []

    # ⭐ prompts.json 파일을 최초 1번만 읽어 캐싱
    if "prompts" not in st.session_state:
        st.session_state["prompts"] = load_prompts()

    debug_block("INIT SESSION", [
        f"FIRST_INIT: {first_init}",
        f"state: {st.session_state['state']}",
        f"substep: {st.session_state['substep']}",
        f"downloads_enabled: {st.session_state['downloads_enabled']}",
        f"generated_questions: {st.session_state['generated_questions']}",
        f"prompts_loaded_keys: {list(st.session_state['prompts'].keys())}"
    ])


# -------------------------------------------------
# 턴 단위 파일 저장 (실시간 append) - 데모 1에서 가져옴
# -------------------------------------------------
def append_turn_to_file(role, text):
    current_index = len(st.session_state["messages"]) - 1
    turn_number = (current_index // 2) + 1
    
    log = {
        "session_id": "sess_004",  # 세션 ID는 데모 4를 반영하여 변경
        "timestamp": datetime.now().isoformat(),
        "role": role,
        "text": text,
        "turn": turn_number
    }
    
    # 저장 경로 설정
    log_dir = os.path.join(os.path.dirname(__file__), "data/logs")  # 실제 환경에 맞게 경로 조정
    os.makedirs(log_dir, exist_ok=True)

    log_path = os.path.join(log_dir, "chat_log.jsonl")

    # JSONL 형식 저장
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(log, ensure_ascii=False) + "\n")

    print(f"[FILE_APPEND] role={role}, turn={turn_number}, path={log_path}")


# -------------------------------------------------
# 메시지 추가 (로그 저장 통합)
# -------------------------------------------------
def add_message(role: str, text: str):
    st.session_state["messages"].append({
        "role": role,
        "message": text,
        "timestamp": datetime.now().isoformat()
    })
    
    current_index = len(st.session_state["messages"]) - 1
    turn_number = (current_index // 2) + 1

    debug_block("ADD MESSAGE", [
        f"ROLE: {role}",
        f"TEXT: {text}",
        f"MESSAGES_LEN: {len(st.session_state['messages'])}",
        f"TURN_NUMBER(approx): {turn_number}"
    ])
    
    # 턴 단위 파일 실시간 저장
    append_turn_to_file(role, text)


# -------------------------------------------------
# UI 렌더링
# -------------------------------------------------
def render_chat_messages():
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
        bot_msg = msgs[i] if msgs[i]["role"] == "bot" else None
        user_msg = msgs[i+1] if i+1 < len(msgs) and msgs[i+1]["role"] == "user" else None

        if bot_msg:
            rows.append({
                "session_id": session_id, "user_id": user_id, "created_at": created_at,
                "chat_type": chat_type, "turn": turn, "role": "bot",
                "text": bot_msg["message"], "timestamp": bot_msg["timestamp"],
            })

        if user_msg:
            rows.append({
                "session_id": session_id, "user_id": user_id, "created_at": created_at,
                "chat_type": chat_type, "turn": turn, "role": "user",
                "text": user_msg["message"], "timestamp": user_msg["timestamp"],
            })
        elif bot_msg:
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
    """

    debug_block("PROCESS FLOW - ENTER", [
        f"RAW user_input: {repr(user_input)}",
        f"CURRENT state: {st.session_state.get('state')}",
        f"CURRENT substep: {st.session_state.get('substep')}"
    ])
    
    # 🔥 0. 대화 종료 후 입력/자동진행 완전 차단
    if st.session_state["state"] == 3 and st.session_state["substep"] == 6:
        debug_block("PROCESS FLOW - END STATE", [
            "state=3 & substep=6 → 종료 상태, 추가 처리 없음"
        ])
        return

    state = st.session_state["state"]
    sub = st.session_state["substep"]

    # GPT 답변 중(sub=1,3,5)에 들어온 유저 입력은 무시
    if user_input and sub not in [2, 4, 6]:
        debug_block("PROCESS FLOW - IGNORE USER INPUT", [
            f"sub={sub} (GPT 자동 발화 턴) 이므로, user_input 무시"
        ])
        return

    # -------------------------------------------------
    # 1. 유저 입력 처리 (sub 2, 4, 6)
    # -------------------------------------------------
    if user_input:
        debug_block("PROCESS FLOW - USER INPUT HANDLING", [
            f"state={state}, substep={sub}",
            f"user_input: {user_input}"
        ])

        add_message("user", user_input)

        if sub in [2, 4]:
            st.session_state["substep"] += 1
            debug_block("PROCESS FLOW - MOVE TO NEXT GPT TURN", [
                f"NEXT substep: {st.session_state['substep']}"
            ])
            st.rerun()

        elif sub == 6:
            if state < 3:
                st.session_state["state"] += 1
                st.session_state["substep"] = 1
                debug_block("PROCESS FLOW - MOVE TO NEXT STATE", [
                    f"NEXT state: {st.session_state['state']}",
                    f"RESET substep: {st.session_state['substep']}"
                ])
                st.rerun()
            else:
                debug_block("PROCESS FLOW - FINAL USER INPUT AT END", [
                    "state=3 & substep=6 에서 user_input 처리 후 종료"
                ])
                return

        return  # user_input 처리 종료

    # -------------------------------------------------
    # 2. GPT/RULE 자동 발화 처리 (user_input == None일 때)
    # -------------------------------------------------
    
    # S1 활동묻기
    if state == 1:
        if sub == 1:  # S1-1: 룰베이스 고정 질문 (첫 로딩 시점)
            debug_block("FSM AUTO BOT - S1 SUB1", [
                "RULE_QUESTION[1] 발화"
            ])
            add_message("bot", RULE_QUESTIONS[1])
            st.session_state["substep"] = 2
            debug_block("FSM TRANSITION", [
                "state=1 유지, substep 1 → 2"
            ])
            st.rerun()
        
        if sub == 3:  # S1-3: GPT 공감 1턴
            last = st.session_state["messages"][-1]["message"]
            debug_block("FSM AUTO BOT - S1 SUB3", [
                f"LAST USER MSG: {last}"
            ])
            bot_msg = generate_empathy_free_question(last, 1, 1)
            add_message("bot", bot_msg)
            st.session_state["substep"] = 4
            debug_block("FSM TRANSITION", [
                "state=1 유지, substep 3 → 4"
            ])
            st.rerun()
        
        if sub == 5:  # S1-5: GPT 공감 2턴
            last = st.session_state["messages"][-1]["message"]
            debug_block("FSM AUTO BOT - S1 SUB5", [
                f"LAST USER MSG: {last}"
            ])
            bot_msg = generate_empathy_free_question(last, 1, 2)
            add_message("bot", bot_msg)
            st.session_state["substep"] = 6
            debug_block("FSM TRANSITION", [
                "state=1 유지, substep 5 → 6"
            ])
            st.rerun()

    # S2 기억회상
    elif state == 2:
        if sub == 1:  # S2-1: GPT 공감 + 고정 질문 (S1 종료 후)
            prev_answer = st.session_state["messages"][-1]["message"]
            fixed = RULE_QUESTIONS[2]
            debug_block("FSM AUTO BOT - S2 SUB1", [
                f"PREV_ANSWER: {prev_answer}",
                f"FIXED_QUESTION: {fixed}"
            ])
            bot_msg = generate_empathy_rule_question(prev_answer, 2, fixed)
            add_message("bot", bot_msg)
            st.session_state["substep"] = 2
            debug_block("FSM TRANSITION", [
                "state=2 유지, substep 1 → 2"
            ])
            st.rerun()
            
        if sub == 3:  # S2-3: GPT 공감 1턴
            last = st.session_state["messages"][-1]["message"]
            debug_block("FSM AUTO BOT - S2 SUB3", [
                f"LAST USER MSG: {last}"
            ])
            bot_msg = generate_empathy_free_question(last, 2, 1)
            add_message("bot", bot_msg)
            st.session_state["substep"] = 4
            debug_block("FSM TRANSITION", [
                "state=2 유지, substep 3 → 4"
            ])
            st.rerun()
            
        if sub == 5:  # S2-5: GPT 공감 2턴
            last = st.session_state["messages"][-1]["message"]
            debug_block("FSM AUTO BOT - S2 SUB5", [
                f"LAST USER MSG: {last}"
            ])
            bot_msg = generate_empathy_free_question(last, 2, 2)
            add_message("bot", bot_msg)
            st.session_state["substep"] = 6
            debug_block("FSM TRANSITION", [
                "state=2 유지, substep 5 → 6"
            ])
            st.rerun()

    # S3 마무리
    elif state == 3:
        if sub == 1:  # S3-1: GPT 공감 + 고정 질문 (S2 종료 후)
            prev_answer = st.session_state["messages"][-1]["message"]
            fixed = RULE_QUESTIONS[3]
            debug_block("FSM AUTO BOT - S3 SUB1", [
                f"PREV_ANSWER: {prev_answer}",
                f"FIXED_QUESTION: {fixed}"
            ])
            bot_msg = generate_empathy_rule_question(prev_answer, 3, fixed)
            add_message("bot", bot_msg)
            st.session_state["substep"] = 2
            debug_block("FSM TRANSITION", [
                "state=3 유지, substep 1 → 2"
            ])
            st.rerun()
            
        if sub == 3:  # S3-3: GPT 공감 1턴
            last = st.session_state["messages"][-1]["message"]
            debug_block("FSM AUTO BOT - S3 SUB3", [
                f"LAST USER MSG: {last}"
            ])
            bot_msg = generate_empathy_free_question(last, 3, 1)
            add_message("bot", bot_msg)
            st.session_state["substep"] = 4
            debug_block("FSM TRANSITION", [
                "state=3 유지, substep 3 → 4"
            ])
            st.rerun()
            
        if sub == 5:  # S3-5: GPT 공감 2턴 (마무리 발화)
            last = st.session_state["messages"][-1]["message"]
            debug_block("FSM AUTO BOT - S3 SUB5 (ENDING)", [
                f"LAST USER MSG: {last}"
            ])
            bot_msg = generate_empathy_ending_message(last)
            add_message("bot", bot_msg)
            st.session_state["substep"] = 6
            debug_block("FSM TRANSITION", [
                "state=3 유지, substep 5 → 6 (END STATE CANDIDATE)"
            ])
            st.rerun()


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
            <div style='font-size: 24px; font-weight: 500; margin-top: -5px;'>(활동 마무리 챗봇)</div>
        </div>
    """, unsafe_allow_html=True)

    # 모델명 표기
    st.markdown(
        f"<div style='text-align:right; color:#888; font-size:14px;'>🔮 model: {MODEL_NAME}</div>",
        unsafe_allow_html=True
    )

    init_session()

    # 1. 렌더링 (이전 세션 상태)
    render_chat_messages()

    state = st.session_state["state"]
    sub = st.session_state["substep"]
    downloads_enabled = st.session_state.get("downloads_enabled", False)

    # --- 입력 가능 substep 정의: 유저 입력 턴만 가능 ---
    can_user_input = (sub in [2, 4, 6])

    # --- S3 종료(sub=6)면 입력창 숨김 ---
    if state == 3 and sub == 6:
        can_user_input = False

    # 항상 기본값 먼저 선언 (오류 방지)
    user_input = None  

    if can_user_input:
        raw_input = st.chat_input("봉봉에게 마음을 이야기해줘 😊")

        if raw_input:  # 입력이 들어온 경우
            # 공백 제외 기준 글자 수
            char_count = len(raw_input.replace(" ", "").replace("\n", ""))

            # 개발자 터미널 로그
            print(f"[USER INPUT RECEIVED] length={char_count} chars (공백 제외)")

            # 200자 초과 시 자동 자르기
            if char_count > 200:
                # 앞에서부터 200글자만 남김
                trimmed = raw_input.replace(" ", "").replace("\n", "")[:200]
                user_input = trimmed
                print(f"[TRIMMED] Input exceeded 200 chars → trimmed to 200.")
            else:
                user_input = raw_input

    else:
        user_input = None


    process_flow(user_input)
    
    # 4. 저장 영역 (S3-5 → substep=6 이후 처리)
    state = st.session_state["state"]
    sub = st.session_state["substep"]
    downloads_enabled = st.session_state.get("downloads_enabled", False)

    # (1) 아직 다운로드 버튼 누르기 전 → 중앙에 "대화 저장" 버튼만 표시
    if state == 3 and sub == 6 and not downloads_enabled:

        st.markdown("<div style='height:25px;'></div>", unsafe_allow_html=True)

        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            save_click = st.button("마무리 완료", use_container_width=True)

        if save_click:
            st.session_state["downloads_enabled"] = True
            debug_block("DOWNLOAD ENABLED", [
                "downloads_enabled set to True"
            ])
            st.rerun()

        st.markdown(
            "<div style='margin-top:35px; margin-bottom:20px; border-top:1px solid #666;'></div>",
            unsafe_allow_html=True
        )

    elif downloads_enabled:
        st.markdown("<hr style='margin-top:35px; margin-bottom:20px;'>", unsafe_allow_html=True)
        st.subheader("📥 대화 저장")

        btn_area = st.columns([0.15, 0.15, 0.7])
        with btn_area[0]:
            save_as_json(disabled=False)
        with btn_area[1]:
            save_as_csv(disabled=False)

        js_code = """
        <script>
            var body = window.parent.document.querySelector('.main');
            body.scrollTop = body.scrollHeight;
        </script>
        """
        components.html(js_code, height=0)


if __name__ == "__main__":
    main()
