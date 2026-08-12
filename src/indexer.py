import json
import os
import re
import tempfile
from pathlib import Path
from typing import Dict, List


WORD_RE = re.compile(r"\w+")


def tokenize(text: str) -> List[str]:
    if not text:
        return []
    return WORD_RE.findall(text.lower())


def _write_block(block_index: Dict[str, set], path: str) -> None:
    # Convert sets to sorted lists for JSON serialization
    serial = {term: sorted(list(docs)) for term, docs in block_index.items()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serial, f, ensure_ascii=False)


def spimi_index(docs: Dict[str, str], block_size: int = 500, temp_dir: str | None = None) -> Dict[str, List[str]]:
    """Build an inverted index using a SPIMI-like blocking strategy.

    Args:
        docs: mapping of doc_id -> text
        block_size: number of documents per in-memory block
        temp_dir: optional directory to write block files to (created if missing)

    Returns:
        merged inverted index: term -> sorted list of doc_ids
    """
    if temp_dir is None:
        temp_dir = tempfile.mkdtemp(prefix="spimi_")
    else:
        Path(temp_dir).mkdir(parents=True, exist_ok=True)

    block_files = []
    items = list(docs.items())
    for i in range(0, len(items), block_size):
        block = items[i : i + block_size]
        block_index: Dict[str, set] = {}
        for doc_id, text in block:
            for token in tokenize(text):
                block_index.setdefault(token, set()).add(str(doc_id))

        block_path = os.path.join(temp_dir, f"block_{i // block_size}.json")
        _write_block(block_index, block_path)
        block_files.append(block_path)

    # Merge blocks
    merged: Dict[str, set] = {}
    for bf in block_files:
        with open(bf, "r", encoding="utf-8") as f:
            part = json.load(f)
        for term, doc_list in part.items():
            merged.setdefault(term, set()).update(map(str, doc_list))

    # Convert to sorted lists
    final_index: Dict[str, List[str]] = {t: sorted(list(d)) for t, d in merged.items()}
    return final_index


def save_index(index: Dict[str, List[str]], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False)


def load_index(path: str) -> Dict[str, List[str]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def boolean_search(index: Dict[str, List[str]], query: str) -> List[str]:
    """Simple AND boolean search: returns docs that contain all query terms."""
    tokens = [t for t in tokenize(query) if t in index]
    if not tokens:
        return []

    sets = [set(index[t]) for t in tokens]
    result = set.intersection(*sets)
    return sorted(list(result))


if __name__ == "__main__":
    sample = {
        "1": "Apple banana orange",
        "2": "Banana carrot",
        "3": "Apple carrot banana",
    }
    idx = spimi_index(sample, block_size=2)
    print("Index terms:", list(idx.keys()))
    print("Search 'apple banana' ->", boolean_search(idx, "apple banana"))
