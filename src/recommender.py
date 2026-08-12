import pandas as pd


def load_interactions(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["user_id"] = df["user_id"].astype(str)
    df["item_id"] = df["item_id"].astype(str)
    df["interaction_type"] = df["interaction_type"].astype(str)
    return df


def build_user_scores(interactions: pd.DataFrame) -> pd.DataFrame:
    weights = {
        "purchase": 3.0,
        "rating": 2.0,
        "click": 1.0,
        "view": 0.5,
    }
    interactions = interactions.copy()
    interactions["score"] = interactions["interaction_type"].map(weights).fillna(0.5)
    interactions["rating"] = pd.to_numeric(interactions["rating"], errors="coerce")
    interactions.loc[interactions["rating"].notna(), "score"] += interactions["rating"] * 0.5
    user_scores = (
        interactions.groupby(["user_id", "item_id"])["score"]
        .sum()
        .reset_index()
        .sort_values(["user_id", "score"], ascending=[True, False])
    )
    return user_scores


def personalize_candidates(user_id: str, candidate_items: list, user_scores: pd.DataFrame) -> list:
    user_history = user_scores[user_scores["user_id"] == user_id]
    if user_history.empty:
        return candidate_items

    history_scores = user_history.set_index("item_id")["score"].to_dict()
    ranked = []
    for item_id, base_score in candidate_items:
        history_score = history_scores.get(str(item_id), 0.0)
        final_score = base_score + history_score
        ranked.append((item_id, final_score))

    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test the personality recommender logic.")
    parser.add_argument("--interactions", default="data/user_interactions.csv")
    parser.add_argument("--user", default="u001")
    args = parser.parse_args()

    interactions = load_interactions(args.interactions)
    user_scores = build_user_scores(interactions)
    sample_candidates = [(str(i), 1.0) for i in range(1, 21)]
    ranked = personalize_candidates(args.user, sample_candidates, user_scores)
    print(ranked[:10])
