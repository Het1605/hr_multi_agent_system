import os
from typing import List
from config.settings import KNOWLEDGE_PATH


def load_knowledge_files() -> List[str]:
    documents = []

    for filename in os.listdir(KNOWLEDGE_PATH):
        file_path = os.path.join(KNOWLEDGE_PATH, filename)

        if not os.path.isfile(file_path):
            continue

        if filename.endswith(".txt") or filename.endswith(".md"):
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    documents.append(content)

    return documents