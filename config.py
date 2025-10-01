import os
from pathlib import Path

DATA_DIR = Path(os.getenv("EVIDENCE_DATA_DIR", "data"))
INDEX_DIR = Path(os.getenv("EVIDENCE_INDEX_DIR", str(DATA_DIR / "index")))
INDEX_PATH = Path(os.getenv("EVIDENCE_INDEX_PATH", str(INDEX_DIR / "faiss.index")))
META_PATH = Path(os.getenv("EVIDENCE_META_PATH", str(DATA_DIR / "meta.jsonl")))
ALLOWED_EXTENSIONS = {"txt", "pdf", "docx"}
MAX_FILE_SIZE_MB = 10
MAX_FILE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
MAX_PDF_PAGES = 50
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
API_PREFIX = "/api/v1"
DEFAULT_TOP_K = 4

INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
META_PATH.parent.mkdir(parents=True, exist_ok=True)
