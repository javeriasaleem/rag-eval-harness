"""
spot_check.py - one-off diagnostic, not part of the final project.

Runs a handful of REAL questions from our ground-truth test_set.json
through the actual pipeline, and prints whether retrieval found the
expected source - a manual, human-eyeball preview of what the full
evaluation harness (Step 6) will do automatically and at scale.

WHY WE PICK THESE SPECIFIC QUESTION IDS:
- q01/q02: a distractor pair (FastAPI vs Starlette background tasks) -
  tests whether retrieval can tell two similar-but-different libraries apart
- q06: an exact_fact question (JWT algorithm) - we already know this works
- q11: a no_answer_in_corpus question - tests correct abstention
- q20: a multi_hop question (Starlette vs FastAPI HTTPException) - the
  hardest category, good to see how it behaves even if not perfect yet
"""

import json
import os

from retrieval import answer_question

TEST_IDS_TO_CHECK = ["q01", "q02", "q06", "q11", "q20"]


def load_test_set():
    path = os.path.join(os.path.dirname(__file__), "..", "test_set.json")
    with open(path, "r") as f:
        return json.load(f)


def main():
    test_set = load_test_set()
    questions_by_id = {q["id"]: q for q in test_set}

    for qid in TEST_IDS_TO_CHECK:
        q = questions_by_id[qid]
        print(f"\n{'=' * 60}")
        print(f"[{qid}] ({q['category']}) {q['question']}")
        print(f"Expected source: {q['expected_source']}")

        result = answer_question(q["question"])

        print(f"Retrieved sources: {result['sources']}")
        print(f"Answer: {result['answer']}")


if __name__ == "__main__":
    main()