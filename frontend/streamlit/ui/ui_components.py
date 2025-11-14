import streamlit as st


def chat_container():
    """대화 기록 표시 UI"""
    st.subheader("💬 대화 기록")

    if not st.session_state.dialogue:
        st.info("아직 대화가 없습니다. 메시지를 입력해보세요!")
        return

    for item in st.session_state.dialogue:
        st.write(f"👤 **You**: {item['user_answer']}")


def user_input_box():
    """입력창"""
    return st.chat_input("메시지를 입력하세요...")
