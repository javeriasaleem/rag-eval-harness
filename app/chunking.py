"""
chunking.py

Splits our markdown corpus files into small, retrievable "chunks."

WHY THIS FILE EXISTS:
Embedding and retrieving whole documents is too coarse - a question about
one specific fact (e.g. "what JWT algorithm does FastAPI recommend?")
shouldn't have to compete with, or be diluted by, an entire multi-page
document. We split each doc into focused pieces so retrieval can find the
*specific* paragraph that answers a question.

"""

import os
import re

# Why these numbers specifically:
# - 300 tokens is roughly one focused paragraph or two 
# - 50 tokens of overlap (~15%) is enough to catch a sentence, without bloating storage with heavy duplication.

MAX_TOKENS = 300
OVERLAP_TOKENS = 50


_TOKENS_PER_WORD = 1.3


def count_tokens(text: str) -> int:
    word_count = len(text.split())
    return int(word_count * _TOKENS_PER_WORD)


def split_by_headers(text: str) -> list[dict]:
    """
    Splits markdown text into sections along '## Heading' lines.
    Returns a list of {"heading": str, "text": str} dicts.

    WHY: markdown headers are the author's own signal of where one idea
    ends and another begins - far more reliable than a fixed character
    count for finding good split points.
    """
    # Matches lines starting with 1-6 '#' characters (markdown headers)
    header_pattern = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)

    sections = []
    matches = list(header_pattern.finditer(text))

    if not matches:
        # No headers at all in this file - treat the whole thing as one section
        return [{"heading": "(no heading)", "text": text.strip()}]

    # Grab any content BEFORE the first header (e.g. an intro paragraph)
    if matches[0].start() > 0:
        intro = text[: matches[0].start()].strip()
        if intro:
            sections.append({"heading": "(intro)", "text": intro})

    for i, match in enumerate(matches):
        heading = match.group(2).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section_text = text[start:end].strip()
        if section_text:
            sections.append({"heading": heading, "text": section_text})

    return sections


def split_long_section(text: str) -> list[str]:
    """
    If a section is longer than MAX_TOKENS, split it further by paragraphs,
    accumulating paragraphs into chunks up to the limit, with OVERLAP_TOKENS
    of the previous chunk's tail carried into the next chunk.
    """
    
    if count_tokens(text) <= MAX_TOKENS:
        return [text]

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""

    # Convert our token overlap budget into an approximate word count,
    # since our count_tokens() is word-based rather than a real tokenizer.
    overlap_words = int(OVERLAP_TOKENS / _TOKENS_PER_WORD)

    for para in paragraphs:
        candidate = (current + "\n\n" + para).strip() if current else para
        if count_tokens(candidate) > MAX_TOKENS and current:
            chunks.append(current)
            # Carry the tail of the previous chunk forward as overlap
            tail_words = current.split()[-overlap_words:]
            tail_text = " ".join(tail_words)
            current = (tail_text + "\n\n" + para).strip()
        else:
            current = candidate

    if current:
        chunks.append(current)

    return chunks


def chunk_file(filepath: str, library: str) -> list[dict]:
    """
    Full pipeline for one file: read -> split by headers -> split long
    sections further -> attach metadata to every resulting chunk.

    Returns a list of dicts, each ready to be embedded and stored:
    {
        "text": str,
        "source_file": str,      # e.g. "fastapi/background-tasks.md"
        "library": str,          # e.g. "fastapi"
        "heading": str,          # e.g. "Using BackgroundTasks"
        "chunk_index": int,      # position within this file's sections
    }
    """
    with open(filepath, "r", encoding="utf-8") as f:
        raw_text = f.read()

    filename = os.path.basename(filepath)
    source_file = f"{library}/{filename}"

    sections = split_by_headers(raw_text)

    chunks = []
    chunk_index = 0
    for section in sections:
        sub_chunks = split_long_section(section["text"])
        for sub_text in sub_chunks:
            chunks.append({
                "text": sub_text,
                "source_file": source_file,
                "library": library,
                "heading": section["heading"],
                "chunk_index": chunk_index,
            })
            chunk_index += 1

    return chunks


def chunk_corpus(corpus_dir: str) -> list[dict]:
    """
    Walks the whole corpus/ directory (fastapi/, pydantic/, starlette/
    subfolders) and returns every chunk from every file, ready for embedding.
    """
    all_chunks = []
    for library in sorted(os.listdir(corpus_dir)):
        library_path = os.path.join(corpus_dir, library)
        if not os.path.isdir(library_path):
            continue
        for filename in sorted(os.listdir(library_path)):
            if not filename.endswith(".md"):
                continue
            filepath = os.path.join(library_path, filename)
            all_chunks.extend(chunk_file(filepath, library))

    return all_chunks


if __name__ == "__main__":
    # Quick manual sanity check when running this file directly -
    # not part of the API, just for us to eyeball the output right now.
    corpus_path = os.path.join(os.path.dirname(__file__), "..", "corpus")
    chunks = chunk_corpus(corpus_path)
    print(f"Total chunks produced: {len(chunks)}")
    print(f"Average tokens per chunk: {sum(count_tokens(c['text']) for c in chunks) / len(chunks):.1f}")
    print("\n--- Sample chunk ---")
    sample = chunks[10]
    print(f"Source: {sample['source_file']}")
    print(f"Heading: {sample['heading']}")
    print(f"Tokens: {count_tokens(sample['text'])}")
    print(f"Text preview: {sample['text'][:200]}...")
