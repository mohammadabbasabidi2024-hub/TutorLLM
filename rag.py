"""Hybrid retrieval (BM25 + vector) from Weaviate and answer generation with the tuned model."""
import os

import torch
import weaviate
from peft import PeftModel
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import config
from prompts import SYSTEM_PROMPT, build_user_message


class Retriever:
    def __init__(self) -> None:
        self.embedder = SentenceTransformer(config.EMBED_MODEL)
        self.client = weaviate.connect_to_local()
        self.col = self.client.collections.get(config.WEAVIATE_COLLECTION)

    def search(self, query: str, k: int = config.TOP_K) -> list[dict]:
        qvec = self.embedder.encode(config.QUERY_PREFIX + query, normalize_embeddings=True).tolist()
        res = self.col.query.hybrid(query=query, vector=qvec, alpha=config.HYBRID_ALPHA, limit=k)
        return [{"text": o.properties["text"], "source": o.properties["source"]} for o in res.objects]

    def close(self) -> None:
        self.client.close()


class Tutor:
    def __init__(self) -> None:
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.bfloat16)
        self.tok = AutoTokenizer.from_pretrained(config.BASE_MODEL)
        model = AutoModelForCausalLM.from_pretrained(config.BASE_MODEL,
                                                     quantization_config=bnb, device_map="auto")
        if os.path.isdir(config.ADAPTER_DIR):
            model = PeftModel.from_pretrained(model, config.ADAPTER_DIR)
        self.model = model.eval()

    def generate(self, messages: list[dict], max_new_tokens: int = 600, temperature: float = 0.3) -> str:
        inputs = self.tok.apply_chat_template(messages, add_generation_prompt=True,
                                              return_tensors="pt", return_dict=True).to(self.model.device)
        kwargs = dict(max_new_tokens=max_new_tokens, do_sample=temperature > 0, top_p=0.9)
        if temperature > 0:
            kwargs["temperature"] = temperature
        with torch.no_grad():
            out = self.model.generate(**inputs, **kwargs)
        return self.tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    def answer(self, question: str, passages: list[dict], weak_topics: list[str] | None = None) -> str:
        messages = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_message(question, passages, weak_topics)}]
        return self.generate(messages)
