import os
import time
import uuid
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .embedding_pipeline import EmbeddingPipeline
from .recommender import build_user_scores, load_interactions, personalize_candidates
from .vector_db import VectorDB
from .database import init_db, log_search, log_grade, get_search_history, get_search_results
from .metrics import get_system_metrics
from .indexer import spimi_index, boolean_search, save_index

app = FastAPI(title="Multimodal Search & Recommendation API")

# Enable CORS for Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

EMBEDDING_DIR = Path("data/embeddings")
METADATA_PATH = EMBEDDING_DIR / "metadata.csv"
QUERY_IMAGE_DIR = Path("data/query_images")

# Global variables
user_scores = None
metadata = None
spimi_idx = None
pipeline = None

# We maintain three vector databases
vector_db_text = None
vector_db_image = None
vector_db_fused = None


@app.on_event("startup")
async def startup_event():
    global user_scores, metadata, spimi_idx, pipeline
    global vector_db_text, vector_db_image, vector_db_fused

    # 1. Initialize SQLite Database
    init_db()

    # Create query image folder if it doesn't exist
    QUERY_IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    # 2. Load User Interactions and build scoring matrix
    if Path("data/user_interactions.csv").exists():
        interactions = load_interactions("data/user_interactions.csv")
        user_scores = build_user_scores(interactions)

    # 3. Load metadata and build/load classical SPIMI index
    if METADATA_PATH.exists():
        metadata = pd.read_csv(METADATA_PATH)
        metadata["item_id"] = metadata["item_id"].astype(str)

        # Build SPIMI index from metadata
        docs = {}
        for _, row in metadata.iterrows():
            item_id = str(row["item_id"])
            text = f"{row['title']} {row['description']}"
            docs[item_id] = text

        # Save index to disk
        spimi_idx = spimi_index(docs, block_size=20)
        save_index(spimi_idx, "data/spimi_index.json")

    # 4. Load Vector DBs if embeddings exist
    if EMBEDDING_DIR.exists():
        # Text Embeddings Index (dim = 384)
        text_emb_path = EMBEDDING_DIR / "text_embeddings.npy"
        if text_emb_path.exists():
            text_embs = np.load(text_emb_path)
            vector_db_text = VectorDB(dim=384)
            vector_db_text.build_index(text_embs)

        # Image Embeddings Index (dim = 2048)
        image_emb_path = EMBEDDING_DIR / "image_embeddings.npy"
        if image_emb_path.exists():
            image_embs = np.load(image_emb_path)
            vector_db_image = VectorDB(dim=2048)
            vector_db_image.build_index(image_embs)

        # Fused Embeddings Index (dim = 256)
        fused_emb_path = EMBEDDING_DIR / "fused_embeddings.npy"
        if fused_emb_path.exists():
            fused_embs = np.load(fused_emb_path)
            vector_db_fused = VectorDB(dim=256)
            vector_db_fused.build_index(fused_embs)

    # 5. Initialize the embedding pipeline once globally (saves loading models repeatedly)
    pipeline = EmbeddingPipeline()


@app.post("/search")
async def search(
    user_id: str = Form(...),
    query: str | None = Form(None),
    image: UploadFile | None = File(None),
    search_type: str = Form("vector")  # "vector" or "classical"
):
    if metadata is None:
        return {"error": "Metadata catalog not loaded."}

    # Normalize inputs
    query_str = query.strip() if query else ""

    # Check query validity
    if search_type == "classical" and not query_str:
        return {"error": "Text query is required for classical search."}
    if search_type == "vector" and not query_str and not image:
        return {"error": "Either text query or image query is required for vector search."}

    candidate_items = []  # List of (item_id, score)
    saved_image_path = ""

    # Process search query image if uploaded
    if image is not None:
        # Save image to query image folder
        ext = Path(image.filename).suffix or ".jpg"
        filename = f"query_{int(time.time())}_{uuid.uuid4().hex[:6]}{ext}"
        target_path = QUERY_IMAGE_DIR / filename
        with target_path.open("wb") as buffer:
            buffer.write(image.file.read())
        saved_image_path = str(target_path)

    # 1. CLASSICAL SPIMI BOOLEAN SEARCH
    if search_type == "classical":
        if spimi_idx is None:
            return {"error": "SPIMI inverted index is not initialized."}

        matching_ids = boolean_search(spimi_idx, query_str)
        if not matching_ids:
            candidate_items = []
        else:
            # Score documents based on token overlap percentage
            q_tokens = [t.lower() for t in query_str.split() if t]
            for doc_id in matching_ids:
                matched_row = metadata[metadata["item_id"] == str(doc_id)]
                if not matched_row.empty:
                    row = matched_row.iloc[0]
                    doc_text = (str(row["title"]) + " " + str(row["description"])).lower()
                    overlap = sum(1 for token in q_tokens if token in doc_text)
                    score = overlap / len(q_tokens) if q_tokens else 1.0
                    candidate_items.append((str(doc_id), score))
            # Sort candidate items descending
            candidate_items.sort(key=lambda x: x[1], reverse=True)

    # 2. SEMANTIC VECTOR SEARCH
    else:
        if pipeline is None:
            return {"error": "Embedding models are not initialized."}

        # Scenario A: Multimodal Search (Text + Image)
        if query_str and saved_image_path:
            if vector_db_fused is None:
                return {"error": "Fused embedding index is not loaded."}
            # Extract text embedding
            text_emb = pipeline.text_model.encode([query_str]).to(pipeline.device)
            # Extract image embedding
            image_emb = pipeline.image_model.encode([saved_image_path]).to(pipeline.device)
            # Fuse embeddings
            with torch.no_grad():
                fused_emb = pipeline.fusion_model(text_emb, image_emb).cpu().numpy()

            distances, indices = vector_db_fused.search(fused_emb, k=20)
            candidate_items = [(str(metadata.iloc[idx]["item_id"]), float(dist)) for idx, dist in zip(indices, distances)]

        # Scenario B: Text-only Semantic Search
        elif query_str:
            if vector_db_text is None:
                return {"error": "Text embedding index is not loaded."}
            text_emb = pipeline.text_model.encode([query_str]).numpy()
            distances, indices = vector_db_text.search(text_emb, k=20)
            candidate_items = [(str(metadata.iloc[idx]["item_id"]), float(dist)) for idx, dist in zip(indices, distances)]

        # Scenario C: Image-only Semantic Search
        elif saved_image_path:
            if vector_db_image is None:
                return {"error": "Image embedding index is not loaded."}
            image_emb = pipeline.image_model.encode([saved_image_path]).numpy()
            distances, indices = vector_db_image.search(image_emb, k=20)
            candidate_items = [(str(metadata.iloc[idx]["item_id"]), float(dist)) for idx, dist in zip(indices, distances)]

    # 3. PERSONALIZATION RE-RANKING
    if user_scores is not None and user_id.strip():
        # Ensure candidate items are represented as string item_ids
        cand_list = [(str(item_id), float(score)) for item_id, score in candidate_items]
        ranked = personalize_candidates(user_id, cand_list, user_scores)
    else:
        ranked = candidate_items

    # 4. STORE SEARCH IN DATABASE
    log_search(
        user_id=user_id,
        query_text=query_str,
        image_path=saved_image_path,
        search_type=f"{search_type}_{'fused' if query_str and saved_image_path else ('text' if query_str else 'image')}" if search_type == "vector" else "classical",
        results=ranked[:20]  # Store top 20 in database for analysis
    )

    # 5. POPULATE SEARCH RESULT DETAILS (Top 10)
    base_scores_dict = {str(item_id): float(base_score) for item_id, base_score in candidate_items}
    results = []
    for item_id, score in ranked[:10]:
        matched_rows = metadata[metadata["item_id"] == str(item_id)]
        if not matched_rows.empty:
            row = matched_rows.iloc[0].to_dict()
            row["score"] = float(score)
            base_val = base_scores_dict.get(str(item_id), float(score))
            row["base_score"] = base_val
            row["boost"] = float(score - base_val)
            results.append(row)

    return {
        "user_id": user_id,
        "query": query_str,
        "image_path": saved_image_path,
        "results": results
    }


@app.post("/grade")
async def grade(
    user_id: str = Form(...),
    item_id: str = Form(...),
    rating: int = Form(...),
    query_text: str | None = Form(""),
    image_path: str | None = Form("")
):
    log_grade(
        user_id=user_id,
        item_id=str(item_id),
        query_text=query_text,
        image_path=image_path,
        rating=rating
    )
    return {
        "status": "success",
        "user_id": user_id,
        "item_id": item_id,
        "rating": rating,
        "query_text": query_text,
        "image_path": image_path
    }


@app.get("/metrics")
async def metrics():
    system_metrics = get_system_metrics()
    return system_metrics


@app.get("/search-history")
async def search_history():
    history = get_search_history()
    return {"history": history}


@app.get("/search-results/{search_id}")
async def search_results(search_id: int):
    results = get_search_results(search_id)
    # Populate with metadata details
    detailed_results = []
    for r in results:
        item_id = r["item_id"]
        matched_rows = metadata[metadata["item_id"] == str(item_id)]
        if not matched_rows.empty:
            row = matched_rows.iloc[0].to_dict()
            row["rank"] = r["rank"]
            row["score"] = r["score"]
            detailed_results.append(row)
    return {"search_id": search_id, "results": detailed_results}



@app.get("/user-history/{user_id}")
async def user_history(user_id: str):
    if not Path("data/user_interactions.csv").exists():
        return {"error": "Interactions data file not found."}

    df = pd.read_csv("data/user_interactions.csv")
    df = df.fillna("")
    df["user_id"] = df["user_id"].astype(str)

    user_df = df[df["user_id"] == str(user_id)]
    if user_df.empty:
        return {
            "user_id": user_id,
            "total_interactions": 0,
            "breakdown": {},
            "history": []
        }

    breakdown = user_df["interaction_type"].value_counts().to_dict()
    # Sort interactions by timestamp desc
    history = user_df.sort_values("timestamp", ascending=False).to_dict(orient="records")

    return {
        "user_id": user_id,
        "total_interactions": len(user_df),
        "breakdown": breakdown,
        "history": history
    }
