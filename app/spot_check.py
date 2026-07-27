"""
spot_check.py - one-off diagnostic, not part of the project.
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
