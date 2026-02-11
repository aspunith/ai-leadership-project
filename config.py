"""
Central configuration for the AI Leadership Insight Agent.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
DOCUMENTS_DIR = PROJECT_ROOT / "company_documents"
CHROMA_PERSIST_DIR = PROJECT_ROOT / "vector_store"

# ── LLM / Embedding settings ────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
USE_OPENAI = bool(OPENAI_API_KEY and not OPENAI_API_KEY.startswith("sk-..."))

# OpenAI models (used when USE_OPENAI is True)
OPENAI_CHAT_MODEL = "gpt-4o"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"

# Local / free-tier model (used when USE_OPENAI is False)
LOCAL_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ── Retrieval settings ───────────────────────────────────────────────────
CHUNK_SIZE = 1000          # characters per chunk
CHUNK_OVERLAP = 200        # overlap between consecutive chunks
TOP_K_RESULTS = 5          # number of chunks to retrieve

# ── Answer-generation settings ───────────────────────────────────────────
MAX_ANSWER_TOKENS = 1024
TEMPERATURE = 0.2
