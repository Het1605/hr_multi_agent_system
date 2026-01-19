import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

LLM_MODEL = "gpt-4o"
LLM_TEMPERATURE = 0

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATABASE_PATH = os.path.join(BASE_DIR, "database", "hr.db")

VECTOR_STORE_PATH = os.path.join(BASE_DIR, "vector_store", "faiss_index")

KNOWLEDGE_PATH = os.path.join(BASE_DIR, "knowledge")