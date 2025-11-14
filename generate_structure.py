import os

# 최종 디렉토리 구조 정의
structure = {
    "backend": {
        "config": {},
        "core": {},
        "db": {},
        "gpt": {},
        "prompts": {},
        "schemas": {},
        "services": {},
    },
    "api": {
        "__files__": ["main.py", "dependencies.py"],
        "routers": {
            "__files__": ["chat.py", "session.py"]
        }
    },
    "frontend": {
        "streamlit": {
            "__files__": ["app.py"],
            "core": {},
            "ui": {},
            "assets": {}
        },
        "react": {
            "src": {}
        }
    },
    "data": {
        "logs": {},
        "examples": {}
    },
    "tests": {},
    "__files__": [
        "requirements.txt",
        ".env.example",
        "README.md"
    ]
}


def create_structure(base_path, tree):
    for name, content in tree.items():
        # 파일 처리
        if name == "__files__":
            for file_name in content:
                file_path = os.path.join(base_path, file_name)
                if not os.path.exists(file_path):
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write("")  # 빈 파일 생성
            continue

        # 폴더 생성
        folder_path = os.path.join(base_path, name)
        os.makedirs(folder_path, exist_ok=True)

        # 하위 구조 처리
        if isinstance(content, dict):
            create_structure(folder_path, content)


if __name__ == "__main__":
    base_dir = os.getcwd()  # chatbot-demo 루트에서 실행
    print(f"📁 생성 시작: {base_dir}")
    create_structure(base_dir, structure)
    print("✨ 프로젝트 구조 생성 완료!")
