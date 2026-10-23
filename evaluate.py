"""Exact-answer evaluation on the held-out test split (scheduling, Banker's, paging).
Compares the final numbers / verdict against the C solver's answer. Run with and without the adapter
(set ADAPTER_DIR to a missing folder for the base model) to measure the fine-tuning gain.
Usage: python evaluate.py [max_items]
"""
import collections
import json
import re
import sys

import config
from prompts import SYSTEM_PROMPT, build_user_message
from rag import Tutor

KEY = re.compile(r"Average WT = [\d.]+, Average TAT = [\d.]+|Total page faults = \d+")


def verdict(text: str):
    if "UNSAFE" in text:
        return "UNSAFE"
    return "SAFE" if "SAFE" in text else None


def correct(instruction: str, gold: str, pred: str) -> bool:
    if instruction.startswith("Apply Banker"):
        return verdict(gold) == verdict(pred)
    m = KEY.search(gold)
    return bool(m) and m.group(0) in pred


def main() -> None:
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    tutor = Tutor()
    stats = collections.defaultdict(lambda: [0, 0])
    for i, line in enumerate(open(config.TEST_FILE, encoding="utf-8")):
        if i >= limit:
            break
        r = json.loads(line)
        if not (r["instruction"].startswith(("Apply Banker", "Compute average", "Count page faults"))):
            continue                         # skip MCQs; they need a different check
        pred = tutor.generate([{"role": "system", "content": SYSTEM_PROMPT},
                               {"role": "user", "content": build_user_message(r["instruction"])}],
                              temperature=0.0)
        kind = r["instruction"].split()[0]
        stats[kind][0] += correct(r["instruction"], r["output"], pred)
        stats[kind][1] += 1
    for kind, (ok, n) in stats.items():
        print(f"{kind:10s} {ok}/{n} = {ok / n:.1%}")


if __name__ == "__main__":
    main()
