import os
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, UploadFile
from pydantic import BaseModel

from .embedding_pipeline import EmbeddingPipeline
from .recommender import build_user_scores, load_interactions, personalize_candidates
from .vector_db import VectorDB

app = FastAPI()

EMBEDDING_DIR = Path("data/embeddings")
VECTOR_INDEX_PATH = EMBEDDING_DIR / "vector.index"
METADATA_PATH = EMBEDDING_DIR / "metadata.csv"

user_scores = None
vector_db = None
metadata = None


class SearchRequest(BaseModel):
    user_id: str
    query: str | None = None


@app.on_event("startup")
async def startup_event():
    global user_scores, vector_db, metadata

    if (Path("data/user_interactions.csv").exists()):
        interactions = load_interactions("data/user_interactions.csv")
        user_scores = build_user_scores(interactions)

    if METADATA_PATH.exists():
        metadata = pd.read_csv(METADATA_PATH)

    if EMBEDDING_DIR.exists() and (EMBEDDING_DIR / "fused_embeddings.npy").exists():
        embeddings = np.load(EMBEDDING_DIR / "fused_embeddings.npy")
        vector_db = VectorDB(dim=embeddings.shape[1])
        vector_db.build_index(embeddings)


@app.post("/search")
async def search(user_id: str = Form(...), query: str | None = Form(None)):
    if vector_db is None or metadata is None:
        return {"error": "Embeddings not generated or index not loaded."}

    if query is None or query.strip() == "":
        return {"error": "Query text is required."}

    pipeline = EmbeddingPipeline()
    text_embedding = pipeline.text_model.encode([query]).numpy()
    distances, indices = vector_db.search(text_embedding, k=20)

    candidate_items = [(metadata.iloc[idx]["item_id"], float(dist)) for idx, dist in zip(indices, distances)]
    if user_scores is not None:
        ranked = personalize_candidates(user_id, candidate_items, user_scores)
    else:
        ranked = candidate_items

    results = []
    for item_id, score in ranked:
        row = metadata[metadata["item_id"] == int(item_id)].iloc[0].to_dict()
        row["score"] = float(score)
        results.append(row)

    return {"user_id": user_id, "query": query, "results": results[:10]}


@app.post("/grade")
async def grade(item_id: str = Form(...), user_id: str = Form(...), rating: int = Form(...)):
    path = Path("data/ratings_log.csv")
    header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        line = f"{user_id},{item_id},{rating}\n"
        f.write(line)
    return {"status": "success", "item_id": item_id, "user_id": user_id, "rating": rating}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
