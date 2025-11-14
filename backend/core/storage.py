import json
import os
from typing import Any, Dict, Optional

BASE_PATH = "data"


class JSONStorage:

    def __init__(self):
        os.makedirs(BASE_PATH, exist_ok=True)
        os.makedirs(os.path.join(BASE_PATH, "sessions"), exist_ok=True)
        os.makedirs(os.path.join(BASE_PATH, "state"), exist_ok=True)
        os.makedirs(os.path.join(BASE_PATH, "users"), exist_ok=True)
        os.makedirs(os.path.join(BASE_PATH, "logs"), exist_ok=True)

    # -------------------------------
    # 1) 세션 저장/조회
    # -------------------------------
    def save_session(self, session_id: str, data: Dict[str, Any]):
        path = os.path.join(BASE_PATH, "sessions", f"{session_id}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        path = os.path.join(BASE_PATH, "sessions", f"{session_id}.json")
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    # -------------------------------
    # 2) 상태 저장 (S1/S2/S3/S4)
    # -------------------------------
    def save_state(self, session_id: str, state: Dict[str, Any]):
        path = os.path.join(BASE_PATH, "state", f"{session_id}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    def load_state(self, session_id: str):
        path = os.path.join(BASE_PATH, "state", f"{session_id}.json")
        if not os.path.exists(path):
            return {"current_stage": "S1", "turn": 1}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    # -------------------------------
    # 3) 로그 append (jsonl)
    # -------------------------------
    def append_log(self, record: Dict[str, Any]):
        log_path = os.path.join(BASE_PATH, "logs", "chat_log.jsonl")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
