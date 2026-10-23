# TutorLLM (code skeleton)

Pipeline: C generator -> dataset -> LoRA fine-tune -> RAG (Weaviate hybrid search) -> FastAPI -> student profile.

## Setup
1. `pip install -r requirements.txt`
2. Start Weaviate locally (Docker): `docker run -p 8080:8080 -p 50051:50051 cr.weaviate.io/semitechnologies/weaviate:latest`
3. Put course files (.pdf/.txt/.md) in `data/corpus/` (only material you have the right to use).
4. Generate data with the C program (`gcc -O2 -o gen gen.c && ./gen 1200 42 > raw.jsonl`), or use the included `data/v2_*.jsonl`.

## Run
```
python ingest.py                 # build the vector index
python finetune.py               # train the LoRA adapter (GPU required)
python evaluate.py 100           # exact-answer accuracy on the held-out split
uvicorn app:app --port 8000      # serve the API
```

## Try it
```
curl -X POST localhost:8000/ask -H 'content-type: application/json' \
  -d '{"student_id":"24BCT0229","question":"Explain Banker\'s algorithm"}'
```

## Status
Written to follow the project proposal. Not trained or run end to end yet (needs a GPU, model access
and a Weaviate instance). Library APIs (trl, weaviate-client) change between versions: pin the versions
you test with.
