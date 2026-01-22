from typing import List
from pathlib import Path
import shutil

from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

from tools.file_loader import load_knowledge_files


VECTOR_STORE_PATH = Path("vector_store/faiss_index")

_embeddings = OpenAIEmbeddings()
_vector_store = None


def build_vector_store():
    # Force rebuild: Remove existing index if it exists
    if VECTOR_STORE_PATH.exists():
        shutil.rmtree(VECTOR_STORE_PATH)

    docs = load_knowledge_files()

    if not docs:
        raise ValueError("No knowledge documents found.")

    # Rule 3: Robust Chunking (200-300 tokens)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=30,
        separators=["\n## ", "\n\n", "\n", " ", ""],
        is_separator_regex=False
    )

    chunks = []
    for text in docs:
        chunks.extend(splitter.create_documents([text]))

    VECTOR_STORE_PATH.mkdir(parents=True, exist_ok=True)

    vector_store = FAISS.from_documents(chunks, _embeddings)
    vector_store.save_local(VECTOR_STORE_PATH)
    
    return vector_store


def load_vector_store():
    global _vector_store

    if _vector_store is None:
        if VECTOR_STORE_PATH.exists():
            try:
                _vector_store = FAISS.load_local(
                    VECTOR_STORE_PATH,
                    _embeddings,
                    allow_dangerous_deserialization=True,
                )
            except Exception:
                # If loading fails (e.g. dimensions mismatch), rebuild
                _vector_store = build_vector_store()
        else:
            _vector_store = build_vector_store()

    return _vector_store


def similarity_search(query: str, k: int = 4, threshold: float = 0.75) -> List[str]:
    """
    Search for relevant policy chunks with a strict similarity score threshold.
    Returns ONLY chunks that meet the threshold (0 <= score <= 1 for cosine similarity).
    Note: FAISS Euclidean distance might need conversion or using specific distance metric.
    Commonly OpenAI embeddings + FAISS uses Euclidean (L2).
    However, langchain FAISS wrapper usually normalizes vectors for cosine similarity equivalent.
    
    LangChain FAISS `similarity_search_with_score` returns L2 distance by default.
    Lower is better. 
    BUT: If using OpenAI embeddings (normalized), L2 distance = 2 * (1 - cosine_similarity).
    So:
    Cosine Sim = 0.7  =>  1 - 0.7 = 0.3  =>  2 * 0.3 = 0.6 (L2 Distance)
    We want Similarity >= 0.7, so L2 Distance <= 0.6.
    
    Let's use `similarity_search_with_relevance_scores` which handles normalization usually.
    LangChain FAISS implementation normalized relevance score to (0, 1].
    """
    store = load_vector_store()
    
    # relevance_score_fn is default for OpenAI/Cosine in newer langchain versions
    results = store.similarity_search_with_relevance_scores(query, k=k)
    
    # Filter by threshold
    # Rule 4: Apply similarity threshold (0.75 based on testing)
    valid_chunks = []
    for doc, score in results:
        if score >= threshold:
            valid_chunks.append(doc.page_content)
            
    return valid_chunks