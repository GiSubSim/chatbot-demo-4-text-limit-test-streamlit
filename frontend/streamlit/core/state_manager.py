import streamlit as st
from datetime import datetime
import uuid


def init_session():
    """세션에 기본값 설정"""
    if "user_id" not in st.session_state:
        st.session_state.user_id = str(uuid.uuid4())

    if "chat_type" not in st.session_state:
        st.session_state.chat_type = "rule_based"

    if "dialogue" not in st.session_state:
        st.session_state.dialogue = []


def add_dialogue(question: str, answer: str):
    """대화 한 턴 추가"""
    st.session_state.dialogue.append({
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "user_answer": answer
    })
