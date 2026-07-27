"""
repair_faithfulness.py - one-off repair script.

Re-runs ONLY the faithfulness judging step for questions already saved in
evaluation_progress.json, using the fixed JSON-extraction logic - without
re-calling generate_answer() again, since we already have the real saved
answer text.
"""

import json
import os

from retrieval import retrieve
from embedding import get_client
from evaluation import judge_faithfulness, RESULTS_PATH

client = get_client()

with open(RESULTS_PATH, "r") as f:
    details = json.load(f)

for d in details:
    if d["faithful"] is not None:
        continue  # already has a real verdict, skip

    print(f"Re-judging {d['id']}...")
    retrieved_chunks = retrieve(client, d["question"], top_k=5)

    faithfulness = judge_faithfulness(client, d["question"], d["answer"], retrieved_chunks)

    d["faithful"] = faithfulness["faithful"]
    d["faithfulness_reasoning"] = faithfulness["reasoning"]

    with open(RESULTS_PATH, "w") as f:
        json.dump(details, f, indent=2)

print("\nDone repairing. Re-run evaluation.py's summary logic to see corrected scores.")