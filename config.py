"""Central settings. Override with environment variables."""
import os

# Proposal uses Llama 3.3 70B. Default here is an 8B model so it fits one GPU;
# set BASE_MODEL=meta-llama/Llama-3.3-70B-Instruct on 2x A100 80GB.
BASE_MODEL = os.getenv("BASE_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
ADAPTER_DIR = os.getenv("ADAPTER_DIR", "adapters/tutorllm-lora")

EMBED_MODEL = "BAAI/bge-large-en"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

WEAVIATE_COLLECTION = "OSChunk"
CORPUS_DIR = os.getenv("CORPUS_DIR", "data/corpus")   # textbook, notes, lab docs (.pdf/.txt/.md)
CHUNK_WORDS = 220
CHUNK_OVERLAP = 40
TOP_K = 5
HYBRID_ALPHA = 0.5        # 0 = pure keyword (BM25), 1 = pure vector

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///tutorllm.db")  # use postgresql://... in production

TRAIN_FILE = os.getenv("TRAIN_FILE", "data/v2_train.jsonl")
TEST_FILE = os.getenv("TEST_FILE", "data/v2_test.jsonl")
