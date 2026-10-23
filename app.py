"""FastAPI backend: /ask (RAG + personalised answer), /quiz-result (update profile), /generate/mcq.
Run: uvicorn app:app --host 0.0.0.0 --port 8000
"""
import json
import re
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel
from sqlalchemy import String, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

import config
from prompts import SYSTEM_PROMPT, detect_topic
from rag import Retriever, Tutor

engine = create_engine(config.DATABASE_URL)


class Base(DeclarativeBase):
    pass


class TopicStat(Base):
    __tablename__ = "topic_stats"
    id: Mapped[int] = mapped_column(primary_key=True)
    student_id: Mapped[str] = mapped_column(String(64), index=True)
    topic: Mapped[str] = mapped_column(String(64))
    attempts: Mapped[int] = mapped_column(default=0)
    correct: Mapped[int] = mapped_column(default=0)


state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(engine)
    state["retriever"] = Retriever()
    state["tutor"] = Tutor()
    yield
    state["retriever"].close()


app = FastAPI(title="TutorLLM", lifespan=lifespan)


def weak_topics(student_id: str, min_attempts: int = 3, threshold: float = 0.6) -> list[str]:
    with Session(engine) as s:
        rows = s.scalars(select(TopicStat).where(TopicStat.student_id == student_id)).all()
    return [r.topic for r in rows if r.attempts >= min_attempts and r.correct / r.attempts < threshold]


class AskIn(BaseModel):
    student_id: str
    question: str


class QuizIn(BaseModel):
    student_id: str
    topic: str
    correct: bool


class McqIn(BaseModel):
    topic: str
    difficulty: str = "medium"
    n: int = 3


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/ask")
def ask(body: AskIn):
    t0 = time.time()
    passages = state["retriever"].search(body.question)
    weak = weak_topics(body.student_id)
    reply = state["tutor"].answer(body.question, passages, weak)
    return {"answer": reply, "topic": detect_topic(body.question),
            "sources": [p["source"] for p in passages],
            "weak_topics": weak, "latency_s": round(time.time() - t0, 2)}   # logged for monitoring


@app.post("/quiz-result")
def quiz_result(body: QuizIn):
    with Session(engine) as s:
        row = s.scalars(select(TopicStat).where(TopicStat.student_id == body.student_id,
                                                TopicStat.topic == body.topic)).first()
        if row is None:
            row = TopicStat(student_id=body.student_id, topic=body.topic, attempts=0, correct=0)
            s.add(row)
        row.attempts += 1
        row.correct += int(body.correct)
        s.commit()
    return {"ok": True}


@app.post("/generate/mcq")
def generate_mcq(body: McqIn):
    passages = state["retriever"].search(body.topic, k=4)
    ctx = "\n\n".join(f"[{p['source']}] {p['text']}" for p in passages)
    prompt = (f"Using only this course context:\n{ctx}\n\n"
              f"Write {body.n} {body.difficulty}-difficulty multiple-choice questions on '{body.topic}'. "
              'Return ONLY a JSON list of objects with keys "question", "options" (4 strings), '
              '"answer" (index 0-3) and "explanation".')
    raw = state["tutor"].generate([{"role": "system", "content": SYSTEM_PROMPT},
                                   {"role": "user", "content": prompt}], max_new_tokens=900, temperature=0.5)
    match = re.search(r"\[.*\]", raw, re.S)
    try:
        return {"questions": json.loads(match.group(0)) if match else [], "raw": None if match else raw}
    except json.JSONDecodeError:
        return {"questions": [], "raw": raw}   # faculty should review generated items before use
