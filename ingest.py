"""Chunk the course corpus, embed with bge-large-en and load it into Weaviate.
Usage: python ingest.py
"""
import glob
import os

import weaviate
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from weaviate.classes.config import Configure, DataType, Property
from weaviate.classes.data import DataObject

import config


def read_file(path: str) -> str:
    if path.lower().endswith(".pdf"):
        return "\n".join((pg.extract_text() or "") for pg in PdfReader(path).pages)
    with open(path, encoding="utf-8", errors="ignore") as f:
        return f.read()


def chunk(text: str, size: int, overlap: int) -> list[str]:
    words = text.split()
    step = size - overlap
    return [" ".join(words[i:i + size]) for i in range(0, max(len(words) - overlap, 1), step)]


def main() -> None:
    files = [p for ext in ("pdf", "txt", "md")
             for p in glob.glob(os.path.join(config.CORPUS_DIR, "**", f"*.{ext}"), recursive=True)]
    if not files:
        raise SystemExit(f"No documents found in {config.CORPUS_DIR}")

    records = []
    for path in files:
        name = os.path.basename(path)
        for i, c in enumerate(chunk(read_file(path), config.CHUNK_WORDS, config.CHUNK_OVERLAP)):
            if c.strip():
                records.append({"text": c, "source": f"{name}#{i}"})
    print(f"{len(files)} files -> {len(records)} chunks")

    model = SentenceTransformer(config.EMBED_MODEL)
    vectors = model.encode([r["text"] for r in records], batch_size=32,
                           normalize_embeddings=True, show_progress_bar=True)

    client = weaviate.connect_to_local()
    try:
        if client.collections.exists(config.WEAVIATE_COLLECTION):
            client.collections.delete(config.WEAVIATE_COLLECTION)
        col = client.collections.create(
            config.WEAVIATE_COLLECTION,
            vectorizer_config=Configure.Vectorizer.none(),   # we supply our own vectors
            properties=[Property(name="text", data_type=DataType.TEXT),
                        Property(name="source", data_type=DataType.TEXT)],
        )
        col.data.insert_many([DataObject(properties=r, vector=v.tolist())
                              for r, v in zip(records, vectors)])
        print("Loaded into Weaviate.")
    finally:
        client.close()


if __name__ == "__main__":
    main()
