import os
from dotenv import load_dotenv


load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR", "./chroma_db")
DOCUMENT_DIR = os.getenv("DOCUMENT_DIR", "./documents")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 100))

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2"
)

TOP_K = int(os.getenv("TOP_K", 5))
LOG_FILE = os.getenv("LOG_FILE", "./logs/query.log")
