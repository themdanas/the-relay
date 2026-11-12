import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PAPERS_DIR = DATA_DIR / "papers"
INDEX_DIR = DATA_DIR / "index"
RESULTS_CSV = BASE_DIR / "results.csv"
TRACES_JSONL = BASE_DIR / "traces.jsonl"


ARVIX_PAPERS = {
    "1706.03762v7": {
        "id": "1706.03762v7",
        "short_title": "Attention Is All You Need",
        "url": "https://arxiv.org/pdf/1706.03762v7.pdf",
        "filename": "attention.pdf",
    },
    "2410.00037v1": {
        "id": "2410.00037v1",
        "short_title": "Moshi",
        "url": "https://arxiv.org/pdf/2410.00037v1.pdf",
        "filename": "moshi.pdf",
    },
    "2507.07955v2": {
        "id": "2507.07955v2",
        "short_title": "Dynamic Chunking",
        "url": "https://arxiv.org/pdf/2507.07955v2.pdf",
        "filename": "dynamic_chunking.pdf",
    },
}

@dataclass(frozen=True)
class RetrievalSetting:
    name: str
    chunk_size_tokens: int
    chunk_overlap_tokens: int
    initial_fetch_k: int
    final_top_k: int
    use_reranker: bool 

# the 3 required setting for table 3
RETRIEVAL_SETTINGS = {
"S1_180tok_k3_norerank": RetrievalSetting(
        name="S1_180tok_k3_norerank",
        chunk_size_tokens=180,
        chunk_overlap_tokens=25,
        initial_fetch_k=3,
        final_top_k=3,
        use_reranker=False,
    ),
    "S2_450tok_k4_rerank": RetrievalSetting(
        name="S2_450tok_k4_rerank",
        chunk_size_tokens=450,
        chunk_overlap_tokens=60,
        initial_fetch_k=10,
        final_top_k=4,
        use_reranker=True,
    ),
    "S3_900tok_k6_norerank": RetrievalSetting(
        name="S3_900tok_k6_norerank",
        chunk_size_tokens=900,
        chunk_overlap_tokens=120,
        initial_fetch_k=6,
        final_top_k=6,
        use_reranker=False,
    ),
}
    
DEFAULT_RETRIEVAL_SETTING = "S2_450tok_k4_rerank"

# LLM & Rate-Limit Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
STANDIN_BASE_URL = "http://127.0.0.1:8000/openai/v1"
MODEL_NAME = "llama-3.1-8b-instant"

# Client patience budget ("people give up sooner than you expect")
USER_PATIENCE_BUDGET_MS = 8000
LOW_TPM_THRESHOLD_RATIO = 0.20