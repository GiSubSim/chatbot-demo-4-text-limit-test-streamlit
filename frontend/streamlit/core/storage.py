import json
import csv
from io import StringIO
import os
import streamlit as st


DATA_DIR = "data/logs"  # JSON DB처럼 쓰는 공간


def save_dialogue_json():
    """세션 대화를 JSON 파일로 저장"""
    os.makedirs(DATA_DIR, exist_ok=True)

    user_id = st.session_state.user_id
    file_path = f"{DATA_DIR}/session_{user_id}.json"

    data = {
        "user_id": user_id,
        "chat_type": st.session_state.chat_type,
        "dialogue": st.session_state.dialogue,
    }

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return file_path


def save_dialogue_csv():
    """대화 내용을 CSV로 변환한 문자열 리턴"""
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(["timestamp", "question", "user_answer"])

    for d in st.session_state.dialogue:
        writer.writerow([d["timestamp"], d["question"], d["user_answer"]])

    return output.getvalue()
