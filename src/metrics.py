import numpy as np
import sqlite3
from typing import Dict, Any, List
from scipy import stats
from . import database


def compute_ndcg_at_k(relevances: List[float], k: int = 10) -> float:
    """Compute Normalized Discounted Cumulative Gain at rank K."""
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
    """Compute Average Precision (AP) for binary relevance."""
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
    """Compute system-wide search relevance benchmarks, average execution times,

    modality contributions, and t-test significance parameters.
    """
    conn = sqlite3.connect(database.DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Get all searches
    cursor.execute("SELECT search_id, user_id, query_text, image_path, search_type, execution_time_ms FROM search_history")
    searches = cursor.fetchall()

    metrics = {
        "vector": {"ndcg": [], "ap": [], "latency": []},
        "classical": {"ndcg": [], "ap": [], "latency": []},
        "overall": {"ndcg": [], "ap": [], "latency": []}
    }

    # Tracking modal contributions in the vector searches
    total_text_weight = 0.0
    total_image_weight = 0.0
    multimodal_count = 0

    for search in searches:
        search_id = search["search_id"]
        query_text = search["query_text"] or ""
        image_path = search["image_path"] or ""
        search_type = search["search_type"]
        latency = search["execution_time_ms"] or 0.0

        group = "vector" if "vector" in search_type else "classical"

        # Log latency
        metrics[group]["latency"].append(latency)
        metrics["overall"]["latency"].append(latency)

        # Get results
        cursor.execute(
            "SELECT item_id, text_similarity, image_similarity FROM search_results WHERE search_id = ? ORDER BY rank ASC",
            (search_id,),
        )
        results_rows = cursor.fetchall()
        results = [row["item_id"] for row in results_rows]

        # Calculate average modality weight for this query
        if group == "vector" and results_rows:
            q_text_w = 0.0
            q_img_w = 0.0
            valid_items = 0
            for r in results_rows:
                ts = r["text_similarity"] or 0.0
                isim = r["image_similarity"] or 0.0
                total = ts + isim
                if total > 0:
                    q_text_w += ts / total
                    q_img_w += isim / total
                    valid_items += 1
            if valid_items > 0:
                total_text_weight += (q_text_w / valid_items)
                total_image_weight += (q_img_w / valid_items)
                multimodal_count += 1

        if not results:
            continue

        # Get grades
        cursor.execute(
            "SELECT item_id, rating FROM grades WHERE query_text = ? AND image_path = ?",
            (query_text, image_path),
        )
        grades_dict = {row["item_id"]: row["rating"] for row in cursor.fetchall()}

        has_grades = any(item_id in grades_dict for item_id in results)
        if not has_grades:
            continue

        relevances = [float(grades_dict.get(item_id, 0)) for item_id in results]
        ndcg = compute_ndcg_at_k(relevances, k=10)
        
        binary_relevances = [grades_dict.get(item_id, 0) >= 3 for item_id in results]
        ap = compute_ap(binary_relevances)

        metrics[group]["ndcg"].append(ndcg)
        metrics[group]["ap"].append(ap)
        metrics["overall"]["ndcg"].append(ndcg)
        metrics["overall"]["ap"].append(ap)

    conn.close()

    # Significance Tests (independent t-test)
    p_val_ndcg = 1.0
    p_val_map = 1.0
    
    v_ndcgs = metrics["vector"]["ndcg"]
    c_ndcgs = metrics["classical"]["ndcg"]
    if len(v_ndcgs) > 1 and len(c_ndcgs) > 1:
        stat_ndcg, p_val_ndcg = stats.ttest_ind(v_ndcgs, c_ndcgs, equal_var=False)
        p_val_ndcg = 1.0 if np.isnan(p_val_ndcg) else float(p_val_ndcg)

    v_aps = metrics["vector"]["ap"]
    c_aps = metrics["classical"]["ap"]
    if len(v_aps) > 1 and len(c_aps) > 1:
        stat_map, p_val_map = stats.ttest_ind(v_aps, c_aps, equal_var=False)
        p_val_map = 1.0 if np.isnan(p_val_map) else float(p_val_map)

    # Modality contribution averages
    avg_text_contrib = 0.5
    avg_image_contrib = 0.5
    if multimodal_count > 0:
        avg_text_contrib = float(total_text_weight / multimodal_count)
        avg_image_contrib = float(total_image_weight / multimodal_count)

    # Compile averages
    results_summary = {}
    for key in ["vector", "classical", "overall"]:
        ndcgs = metrics[key]["ndcg"]
        aps = metrics[key]["ap"]
        latencies = metrics[key]["latency"]
        
        results_summary[key] = {
            "NDCG@10": float(np.mean(ndcgs)) if ndcgs else 0.0,
            "MAP": float(np.mean(aps)) if aps else 0.0,
            "latency_ms": float(np.mean(latencies)) if latencies else 0.0,
            "count": len(ndcgs)
        }

    results_summary["stats"] = {
        "p_value_ndcg": p_val_ndcg,
        "p_value_map": p_val_map,
        "is_ndcg_significant": bool(p_val_ndcg < 0.05),
        "is_map_significant": bool(p_val_map < 0.05)
    }

    results_summary["modalities"] = {
        "text_contribution_pct": float(avg_text_contrib * 100),
        "image_contribution_pct": float(avg_image_contrib * 100)
    }

    return results_summary
