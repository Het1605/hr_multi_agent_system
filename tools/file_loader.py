from pathlib import Path
from typing import List


KNOWLEDGE_DIR = Path("knowledge")


def load_knowledge_files() -> List[str]:
    """
    Load all HR policy and company rule documents.
    """
    documents = []

    for file_path in KNOWLEDGE_DIR.glob("*"):
        if file_path.suffix in [".txt", ".md"]:
            documents.append(file_path.read_text(encoding="utf-8"))

    return documents