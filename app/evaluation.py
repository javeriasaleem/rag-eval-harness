"""
evaluation.py

The actual evaluation harness logic: runs our real test_set.json through
the live RAG pipeline and scores three separate dimensions. This is what
turns the project from "a RAG app" into "a RAG EVALUATION harness."

WHY THREE SEPARATE METRICS, NOT ONE OVERALL "SCORE":
A single blended score would hide WHICH layer is failing. If retrieval is
bad but faithfulness looks fine, that's a completely different bug than
good retrieval with bad faithfulness (model ignoring/misusing correct
context). Keeping them separate makes the report actually diagnostic,
not just a pass/fail badge.
"""

import json
import os
import re

from retrieval import answer_question, GENERATION_MODEL
from embedding import get_client
from api_utils import call_with_retry

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "..", "evaluation_progress.json")


def load_progress() -> list[dict]:
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH, "r") as f:
            return json.load(f)
    return []


def save_progress(details: list[dict]):
    with open(RESULTS_PATH, "w") as f:
        json.dump(details, f, indent=2)


def load_test_set():
    path = os.path.join(os.path.dirname(__file__), "..", "test_set.json")
    with open(path, "r") as f:
        return json.load(f)


def check_retrieval(expected_source, retrieved_sources: list[str]) -> bool | None:
    """
    Did retrieval actually surface the expected source file(s)?

    WHY THIS RETURNS None SOMETIMES (not just True/False):
    For no_answer_in_corpus questions, expected_source is None by design -
    there IS no correct source to check retrieval against. Returning None
    (rather than forcing a True/False) lets our aggregation step correctly
    EXCLUDE these questions from the retrieval-accuracy metric, instead of
    silently miscounting them as retrieval failures.
    """
    if expected_source is None:
        return None

    if isinstance(expected_source, list):
        # multi_hop questions: correct if ANY of the expected sources was retrieved
        return any(src in retrieved_sources for src in expected_source)

    return expected_source in retrieved_sources


def check_abstention(category: str, answer: str) -> bool | None:
    """
    For no_answer_in_corpus questions specifically: did the system say it
    doesn't know, instead of confidently hallucinating an answer?

    WHY WE CHECK FOR THE EXACT PHRASE our own prompt specifies:
    Our SYSTEM_PROMPT in retrieval.py explicitly instructs the model to
    say "Not found in the provided context." when it can't answer. So
    checking for that phrase here isn't a lucky guess - it's verifying the
    model actually followed OUR OWN instruction, which is exactly the
    behavior we engineered for and need to confirm is actually happening.
    """
    if category != "no_answer_in_corpus":
        return None  # not applicable to this question

    return "not found in the provided context" in answer.lower()


# WHY THIS EXACT JUDGE PROMPT DESIGN:
# We ask for STRICT JSON output (not free-form text) so we can reliably
# parse the judge's verdict in code, rather than trying to regex-guess
# what a free-form judge response "meant." Forcing structured output is a
# common, important pattern when using an LLM as a programmatic component
# rather than a chat partner.
FAITHFULNESS_JUDGE_PROMPT = """You are a strict fact-checker. You will be given a QUESTION, some CONTEXT that was retrieved for it, and an ANSWER that was generated.

Your job: determine if the ANSWER is fully supported by the CONTEXT. The answer should not contain any claim that isn't backed by the context.

Respond with ONLY a JSON object, no other text, in this exact format:
{{"faithful": true or false, "reasoning": "one short sentence explaining why"}}

QUESTION: {question}

CONTEXT:
{context}

ANSWER: {answer}

JSON:"""


def judge_faithfulness(client, question: str, answer: str, retrieved_chunks: list[dict]) -> dict:
    """
    Uses Gemini itself as a judge to check whether the generated answer is
    actually supported by the retrieved context - NOT compared against our
    ground-truth answer key, but against what was actually retrieved. This
    distinction matters: it catches generation drifting/hallucinating even
    when retrieval itself succeeded.
    """
    context_text = "\n\n---\n\n".join(c["text"] for c in retrieved_chunks)

    prompt = FAITHFULNESS_JUDGE_PROMPT.format(
        question=question,
        context=context_text,
        answer=answer,
    )

    def do_call():
        response = client.models.generate_content(
            model=GENERATION_MODEL,
            contents=prompt,
        )
        return response.text


   # Same bucket as retrieval.py's generate_answer - BOTH use
    # gemini-3.5-flash and share the same 5 RPM limit, so they must be
    # paced together under one shared "generation" bucket, not separately.
    raw_text = call_with_retry(do_call, bucket="generation", min_interval_seconds=13).strip()

    first_brace = raw_text.find("{")
    last_brace = raw_text.rfind("}")

    if first_brace == -1 or last_brace == -1 or last_brace < first_brace:
        return {"faithful": None, "reasoning": f"No JSON object found in judge response: {raw_text[:200]}"}

    json_candidate = raw_text[first_brace:last_brace + 1]

    try:
        return json.loads(json_candidate)
    except json.JSONDecodeError:
        return {"faithful": None, "reasoning": f"Could not parse judge response: {raw_text[:200]}"}


def run_evaluation() -> dict:
    """
    Runs every question in test_set.json through the live pipeline.
    Resumable: saves progress after every question, and skips any
    question already completed in a previous run - so a quota limit
    only costs you the remaining questions, not a full restart.
    """
    client = get_client()
    test_set = load_test_set()

    details = load_progress()
    completed_ids = {d["id"] for d in details}

    remaining = [q for q in test_set if q["id"] not in completed_ids]
    print(f"{len(completed_ids)} questions already completed, skipping those. "
          f"{len(remaining)} remaining to process now.")

    for q in remaining:
        result = answer_question(q["question"], top_k=5)

        retrieval_hit = check_retrieval(q["expected_source"], result["sources"])
        abstention_correct = check_abstention(q["category"], result["answer"])

        faithfulness = judge_faithfulness(
            client, q["question"], result["answer"], result["retrieved_chunks"]
        )

        details.append({
            "id": q["id"],
            "category": q["category"],
            "question": q["question"],
            "expected_source": q["expected_source"],
            "retrieved_sources": result["sources"],
            "retrieval_hit": retrieval_hit,
            "answer": result["answer"],
            "faithful": faithfulness["faithful"],
            "faithfulness_reasoning": faithfulness["reasoning"],
            "abstention_correct": abstention_correct,
        })

        save_progress(details)
        print(f"  Completed and saved: {q['id']} ({len(details)}/{len(test_set)} total)")

    retrieval_scores = [d["retrieval_hit"] for d in details if d["retrieval_hit"] is not None]
    faithfulness_scores = [d["faithful"] for d in details if d["faithful"] is not None]
    abstention_scores = [d["abstention_correct"] for d in details if d["abstention_correct"] is not None]

    summary = {
        "total_questions": len(details),
        "retrieval_accuracy": sum(retrieval_scores) / len(retrieval_scores) if retrieval_scores else None,
        "faithfulness_rate": sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else None,
        "abstention_accuracy": sum(abstention_scores) / len(abstention_scores) if abstention_scores else None,
    }

    return {"summary": summary, "details": details}


if __name__ == "__main__":
    report = run_evaluation()
    print(json.dumps(report["summary"], indent=2))
    print(f"\n({len(report['details'])} questions evaluated in detail - see returned report for full breakdown)")