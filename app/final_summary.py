import json
import os
from evaluation import RESULTS_PATH

with open(RESULTS_PATH, "r") as f:
    details = json.load(f)

unresolved = [d["id"] for d in details if d["faithful"] is None]
if unresolved:
    print(f"Still {len(unresolved)} unresolved (faithful=null): {unresolved}")
    print("Run repair_faithfulness.py again before trusting this summary.\n")

retrieval_scores = [d["retrieval_hit"] for d in details if d["retrieval_hit"] is not None]
faithfulness_scores = [d["faithful"] for d in details if d["faithful"] is not None]
abstention_scores = [d["abstention_correct"] for d in details if d["abstention_correct"] is not None]

summary = {
    "total_questions": len(details),
    "retrieval_accuracy": sum(retrieval_scores) / len(retrieval_scores) if retrieval_scores else None,
    "faithfulness_rate": sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else None,
    "faithfulness_scored_out_of": len(faithfulness_scores),
    "abstention_accuracy": sum(abstention_scores) / len(abstention_scores) if abstention_scores else None,
}

print(json.dumps(summary, indent=2))

failures = [d for d in details if d["faithful"] is False]
if failures:
    print(f"\n{len(failures)} genuine faithfulness failure(s) found:")
    for d in failures:
        print(f"  {d['id']}: {d['question']}")
        print(f"    Reasoning: {d['faithfulness_reasoning']}")
        