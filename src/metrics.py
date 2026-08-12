import numpy as np
import sqlite3
from typing import Dict, Any, List
from . import database


def compute_ndcg_at_k(relevances: List[float], k: int = 10) -> float:
    """Compute Normalized Discounted Cumulative Gain at rank K.

    Args:
        relevances: List of relevance scores (e.g. 1-5 scale) ordered by rank.
        k: Rank threshold.

    Returns:
        NDCG score between 0.0 and 1.0.
    """
    relevances = relevances[:k]
    if not relevances:
        return 0.0

    # DCG = sum_{i=1}^k ( (2^rel_i - 1) / log2(i + 1) )
    dcg = 0.0
    for idx, rel in enumerate(relevances):
        dcg += (2**rel - 1) / np.log2(idx + 2)

    # Ideal DCG (sort descending)
    ideal_relevances = sorted(relevances, reverse=True)
    idcg = 0.0
    for idx, rel in enumerate(ideal_relevances):
        idcg += (2**rel - 1) / np.log2(idx + 2)

    if idcg == 0.0:
        return 0.0
    return float(dcg / idcg)


def compute_ap(relevances: List[bool]) -> float:
    """Compute Average Precision (AP) for binary relevance.

    Args:
        relevances: List of booleans representing if the item at each rank is relevant.

    Returns:
        AP score between 0.0 and 1.0.
    """
    num_relevant = sum(relevances)
    if num_relevant == 0:
        return 0.0

    ap = 0.0
    relevant_found = 0
    for idx, rel in enumerate(relevances):
        if rel:
            relevant_found += 1
            precision_at_k = relevant_found / (idx + 1)
            ap += precision_at_k

    return float(ap / num_relevant)


def get_system_metrics() -> Dict[str, Any]:
    """Compute system-wide search relevance benchmarks comparing classical vs semantic search.

    Returns:
        Structured dict with metrics split by classical, vector, and overall search types.
    """
    conn = sqlite3.connect(database.DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Get all searches
    cursor.execute("SELECT search_id, user_id, query_text, image_path, search_type FROM search_history")
    searches = cursor.fetchall()

    metrics = {
        "vector": {"ndcg": [], "ap": []},
        "classical": {"ndcg": [], "ap": []},
        "overall": {"ndcg": [], "ap": []}
    }

    for search in searches:
        search_id = search["search_id"]
        query_text = search["query_text"] or ""
        image_path = search["image_path"] or ""
        search_type = search["search_type"]  # e.g., "vector_text", "classical"

        # Categorize
        group = "vector" if "vector" in search_type else "classical"

        # Get results
        cursor.execute(
            "SELECT item_id FROM search_results WHERE search_id = ? ORDER BY rank ASC",
            (search_id,),
        )
        results = [row["item_id"] for row in cursor.fetchall()]

        if not results:
            continue

        # Get grades for this exact query text + image context
        cursor.execute(
            "SELECT item_id, rating FROM grades WHERE query_text = ? AND image_path = ?",
            (query_text, image_path),
        )
        grades_dict = {row["item_id"]: row["rating"] for row in cursor.fetchall()}

        # Skip if no grades are available for any of these results
        has_grades = any(item_id in grades_dict for item_id in results)
        if not has_grades:
            continue

        # Compute metrics
        relevances = [float(grades_dict.get(item_id, 0)) for item_id in results]
        ndcg = compute_ndcg_at_k(relevances, k=10)
        
        binary_relevances = [grades_dict.get(item_id, 0) >= 3 for item_id in results]
        ap = compute_ap(binary_relevances)

        # Store in correct category and overall
        metrics[group]["ndcg"].append(ndcg)
        metrics[group]["ap"].append(ap)
        metrics["overall"]["ndcg"].append(ndcg)
        metrics["overall"]["ap"].append(ap)

    conn.close()

    # Calculate averages
    results_summary = {}
    for key in ["vector", "classical", "overall"]:
        ndcgs = metrics[key]["ndcg"]
        aps = metrics[key]["ap"]
        
        results_summary[key] = {
            "NDCG@10": float(np.mean(ndcgs)) if ndcgs else 0.0,
            "MAP": float(np.mean(aps)) if aps else 0.0,
            "count": len(ndcgs)
        }

    return results_summary
