"""Prompt templates shared by fine-tuning and inference so the model sees one format."""

SYSTEM_PROMPT = (
    "You are TutorLLM, a patient Operating Systems tutor for university students. "
    "Use the provided course context when it is relevant and name the source in square brackets. "
    "Explain step by step, use a short analogy when it helps, and show your working for calculations. "
    "If the context does not contain the answer, say so instead of inventing details. "
    "Never make up numbers: for scheduling, Banker's algorithm and page replacement, "
    "state every assumption (quantum, tie-break rule, preemption) and show each step."
)

TOPIC_KEYWORDS = {
    "scheduling": ["scheduling", "fcfs", "sjf", "round robin", "quantum", "turnaround", "waiting time"],
    "deadlock": ["deadlock", "banker", "safe state", "resource allocation", "need matrix"],
    "paging": ["paging", "page fault", "page replacement", "lru", "fifo", "optimal", "frame", "tlb"],
    "synchronization": ["semaphore", "mutex", "monitor", "race condition", "critical section", "producer", "consumer"],
}


def detect_topic(text: str) -> str | None:
    t = text.lower()
    for topic, words in TOPIC_KEYWORDS.items():
        if any(w in t for w in words):
            return topic
    return None


def build_user_message(question: str, passages: list[dict] | None = None,
                       weak_topics: list[str] | None = None) -> str:
    ctx = "(no context retrieved)"
    if passages:
        ctx = "\n\n".join(f"[{p['source']}] {p['text']}" for p in passages)
    parts = [f"Course context:\n{ctx}"]
    if weak_topics:
        parts.append("Student's weaker topics (explain more simply, with an extra example): "
                     + ", ".join(weak_topics))
    parts.append(f"Question: {question}")
    return "\n\n".join(parts)
