"""Central configuration for the DSPy RAG system.

Everything tunable lives here so the rest of the code reads cleanly and you can
swap models or paths in one place. All defaults are chosen to run comfortably on
a 16 GB Apple Silicon Mac, fully offline once models are pulled.
"""
from pathlib import Path

# --- Paths -------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"                 # the knowledge base (.md files)
CHROMA_DIR = ROOT / "chroma_db"          # persisted vector store
EVAL_DIR = ROOT / "eval"
GOLD_QA_PATH = EVAL_DIR / "qa_gold.json"
RESULTS_DIR = EVAL_DIR / "results"

# --- Local models ------------------------------------------------------------
# Generation: a small instruction-tuned model served by Ollama. It answers
# questions from retrieved context. Speed matters more than strict formatting.
OLLAMA_MODEL = "llama3.2:3b"
OLLAMA_BASE_URL = "http://localhost:11434"

# Judge: a SEPARATE model used only by RAGAS to grade answers. Judging needs
# disciplined, parseable output (JSON verdicts) more than broad knowledge, so
# we use Qwen2.5-3B, which follows instructions far more reliably than the
# generator. This generator-vs-judge split is standard practice in LLM eval.
JUDGE_MODEL = "qwen2.5:3b"

# Retrieval: a small sentence-transformer embedding model. This runs locally
# via sentence-transformers (downloaded once, then offline). Note it is a
# DIFFERENT model from the generator — retrieval and generation are separate
# jobs, and small embedding models are fast and accurate for search.
EMBED_MODEL = "all-MiniLM-L6-v2"

# --- Retrieval settings ------------------------------------------------------
CHUNK_SIZE = 500          # characters per chunk (roughly a paragraph)
CHUNK_OVERLAP = 100       # characters shared between neighboring chunks
TOP_K = 4                 # how many chunks to retrieve per question
COLLECTION_NAME = "acme_handbook"
