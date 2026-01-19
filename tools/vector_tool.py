import os
from typing import List

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from config.settings import VECTOR_STORE_PATH
from tools.file_loader import load_knowledge_files


_embeddings = OpenAIEmbeddings()
_vector_store = None


def build_vector_store():
    global _vector_store

    documents = load_knowledge_files()
    if not documents:
        raise ValueError("No knowledge documents found.")

    _vector_store = FAISS.from_texts(documents, _embeddings)

    os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
    _vector_store.save_local(VECTOR_STORE_PATH)


def load_vector_store():
    global _vector_store

    index_file = os.path.join(VECTOR_STORE_PATH, "index.faiss")

    if not os.path.exists(index_file):
        build_vector_store()
    else:
        _vector_store = FAISS.load_local(
            VECTOR_STORE_PATH,
            _embeddings,
            allow_dangerous_deserialization=True,
        )


def similarity_search(query: str, k: int = 3) -> List[str]:
    global _vector_store

    if _vector_store is None:
        load_vector_store()

    results = _vector_store.similarity_search(query, k=k)
    return [doc.page_content for doc in results]