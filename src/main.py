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
from .database import init_db, log_search, log_grade, get_search_history, get_search_results, seed_initial_grades
from .metrics import get_system_metrics, compute_ndcg_at_k, compute_ap
from .indexer import spimi_index, boolean_search, save_index
from .fusion_model import FusionModel


app = FastAPI(title="Bridging the Modality Gap: Adaptive Feature Dropout and Hybrid Indexing in Multimodal E-Commerce Search API")

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

# Global embedding matrices
text_embeddings = None
image_embeddings = None

# We maintain three vector databases
vector_db_text = None
vector_db_image = None
vector_db_fused = None


@app.on_event("startup")
async def startup_event():
    global user_scores, metadata, spimi_idx, pipeline
    global vector_db_text, vector_db_image, vector_db_fused
    global text_embeddings, image_embeddings

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
        
        # Seed initial query grades for evaluation
        seed_initial_grades(metadata)


    # 4. Load Vector DBs if embeddings exist
    if EMBEDDING_DIR.exists():
        # Text Embeddings Index (dim = 384)
        text_emb_path = EMBEDDING_DIR / "text_embeddings.npy"
        if text_emb_path.exists():
            text_embeddings = np.load(text_emb_path)
            vector_db_text = VectorDB(dim=384)
            vector_db_text.build_index(text_embeddings)

        # Image Embeddings Index (dim = 2048)
        image_emb_path = EMBEDDING_DIR / "image_embeddings.npy"
        if image_emb_path.exists():
            image_embeddings = np.load(image_emb_path)
            vector_db_image = VectorDB(dim=2048)
            vector_db_image.build_index(image_embeddings)

        # Fused Embeddings Index (dim = 256)
        fused_emb_path = EMBEDDING_DIR / "fused_embeddings.npy"
        if fused_emb_path.exists():
            fused_embs = np.load(fused_emb_path)
            vector_db_fused = VectorDB(dim=256)
            vector_db_fused.build_index(fused_embs)

    # 5. Initialize the embedding pipeline once globally (saves loading models repeatedly)
    checkpoint_file = str(EMBEDDING_DIR / "fusion_model.pt")
    pipeline = EmbeddingPipeline(checkpoint_path=checkpoint_file)



@app.post("/search")
async def search(
    user_id: str = Form(...),
    query: str | None = Form(None),
    image: UploadFile | None = File(None),
    search_type: str = Form("vector"),  # "vector" or "classical"
    index_mode: str = Form("fused")     # "fused" (single index) or "split" (split indices)
):
    if metadata is None:
        return {"error": "Metadata catalog not loaded."}

    # Start search latency timer
    start_time = time.time()

    # Normalize inputs
    query_str = query.strip() if query else ""

    # Check query validity
    if search_type == "classical" and not query_str:
        return {"error": "Text query is required for classical search."}
    if search_type == "vector" and not query_str and not image:
        return {"error": "Either text query or image query is required for vector search."}

    candidate_items = []  # List of (item_id, score)
    saved_image_path = ""
    query_text_emb = None
    query_image_emb = None

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
            text_emb_tensor = pipeline.text_model.encode([query_str]).to(pipeline.device)
            query_text_emb = text_emb_tensor.cpu().numpy()
            
            # Extract image embedding
            image_emb_tensor = pipeline.image_model.encode([saved_image_path]).to(pipeline.device)
            query_image_emb = image_emb_tensor.cpu().numpy()
            
            # Fuse embeddings
            with torch.no_grad():
                fused_emb = pipeline.fusion_model(text_emb_tensor, image_emb_tensor).cpu().numpy()

            distances, indices = vector_db_fused.search(fused_emb, k=20)
            candidate_items = [(str(metadata.iloc[idx]["item_id"]), float(dist)) for idx, dist in zip(indices, distances)]

        # Scenario B: Text-only Semantic Search
        elif query_str:
            if index_mode == "fused":
                # Single Index Routing: project query text with zeroed image vector in fused space
                if vector_db_fused is None:
                    return {"error": "Fused embedding index is not loaded."}
                text_emb_tensor = pipeline.text_model.encode([query_str]).to(pipeline.device)
                query_text_emb = text_emb_tensor.cpu().numpy()
                
                # Zero out image embedding tensor
                zero_image_emb = torch.zeros(1, 2048, device=pipeline.device)
                with torch.no_grad():
                    fused_emb = pipeline.fusion_model(text_emb_tensor, zero_image_emb).cpu().numpy()
                
                distances, indices = vector_db_fused.search(fused_emb, k=20)
                candidate_items = [(str(metadata.iloc[idx]["item_id"]), float(dist)) for idx, dist in zip(indices, distances)]
            else:
                # Split Index Routing (Baseline)
                if vector_db_text is None:
                    return {"error": "Text embedding index is not loaded."}
                query_text_emb = pipeline.text_model.encode([query_str]).numpy()
                distances, indices = vector_db_text.search(query_text_emb, k=20)
                candidate_items = [(str(metadata.iloc[idx]["item_id"]), float(dist)) for idx, dist in zip(indices, distances)]

        # Scenario C: Image-only Semantic Search
        elif saved_image_path:
            if index_mode == "fused":
                # Single Index Routing: project query image with zeroed text vector in fused space
                if vector_db_fused is None:
                    return {"error": "Fused embedding index is not loaded."}
                image_emb_tensor = pipeline.image_model.encode([saved_image_path]).to(pipeline.device)
                query_image_emb = image_emb_tensor.cpu().numpy()
                
                # Zero out text embedding tensor
                zero_text_emb = torch.zeros(1, 384, device=pipeline.device)
                with torch.no_grad():
                    fused_emb = pipeline.fusion_model(zero_text_emb, image_emb_tensor).cpu().numpy()
                
                distances, indices = vector_db_fused.search(fused_emb, k=20)
                candidate_items = [(str(metadata.iloc[idx]["item_id"]), float(dist)) for idx, dist in zip(indices, distances)]
            else:
                # Split Index Routing (Baseline)
                if vector_db_image is None:
                    return {"error": "Image embedding index is not loaded."}
                query_image_emb = pipeline.image_model.encode([saved_image_path]).numpy()
                distances, indices = vector_db_image.search(query_image_emb, k=20)
                candidate_items = [(str(metadata.iloc[idx]["item_id"]), float(dist)) for idx, dist in zip(indices, distances)]


    # 3. PERSONALIZATION RE-RANKING
    if user_scores is not None and user_id.strip():
        # Ensure candidate items are represented as string item_ids
        cand_list = [(str(item_id), float(score)) for item_id, score in candidate_items]
        ranked = personalize_candidates(user_id, cand_list, user_scores)
    else:
        ranked = candidate_items

    # Stop latency timer
    execution_time_ms = (time.time() - start_time) * 1000

    # 4. COMPUTE MODAL SIMILARITIES FOR RETRIEVED PRODUCTS
    base_scores_dict = {str(item_id): float(base_score) for item_id, base_score in candidate_items}
    results_to_log = []

    for item_id, score in ranked:
        matched_idx = metadata.index[metadata["item_id"] == str(item_id)]
        text_sim = 0.0
        image_sim = 0.0

        if not matched_idx.empty:
            idx = matched_idx[0]

            # Text similarity
            if query_text_emb is not None and text_embeddings is not None:
                q_text_vec = query_text_emb.reshape(-1)
                item_text_vec = text_embeddings[idx].reshape(-1)
                text_sim = float(np.dot(q_text_vec, item_text_vec))
            elif search_type == "classical":
                text_sim = base_scores_dict.get(str(item_id), 0.0)

            # Image similarity
            if query_image_emb is not None and image_embeddings is not None:
                q_img_vec = query_image_emb.reshape(-1)
                item_img_vec = image_embeddings[idx].reshape(-1)
                image_sim = float(np.dot(q_img_vec, item_img_vec))

        results_to_log.append((item_id, score, text_sim, image_sim))

    # 5. STORE SEARCH IN DATABASE
    log_search(
        user_id=user_id,
        query_text=query_str,
        image_path=saved_image_path,
        search_type=f"{search_type}_{'fused' if query_str and saved_image_path else ('text' if query_str else 'image')}" if search_type == "vector" else "classical",
        results=results_to_log[:20],
        execution_time_ms=execution_time_ms
    )

    # 6. POPULATE SEARCH RESULT DETAILS (Top 10)
    results = []
    for item_id, score, text_sim, image_sim in results_to_log[:10]:
        matched_rows = metadata[metadata["item_id"] == str(item_id)]
        if not matched_rows.empty:
            row = matched_rows.iloc[0].to_dict()
            row["score"] = float(score)
            base_val = base_scores_dict.get(str(item_id), float(score))
            row["base_score"] = base_val
            row["boost"] = float(score - base_val)
            row["text_similarity"] = text_sim
            row["image_similarity"] = image_sim
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


class AblationRequest(BaseModel):
    missing_rate: float


@app.post("/simulate-ablation")
async def simulate_ablation(req: AblationRequest):
    if metadata is None or text_embeddings is None or image_embeddings is None:
        return {"error": "Catalog metadata or pre-computed embeddings are not loaded."}
        
    missing_rate = req.missing_rate
    if not (0.0 <= missing_rate <= 1.0):
        return {"error": "missing_rate must be between 0.0 and 1.0"}
        
    # Simulate missing images in the catalog by zeroing out visual vectors
    np.random.seed(42)
    N = len(metadata)
    
    # Identify random indices to drop visual features
    dropped_mask = np.random.rand(N) < missing_rate
    
    text_t = torch.tensor(text_embeddings, dtype=torch.float32, device=pipeline.device)
    image_t = torch.tensor(image_embeddings, dtype=torch.float32, device=pipeline.device)
    
    image_dropped_t = image_t.clone()
    # Zero out visual embeddings for the selected dropped subset
    image_dropped_t[dropped_mask] = 0.0
    
    # 1. Baseline: Untrained Model (random weights initialized and evaluated on zeroed inputs)
    untrained_model = FusionModel().to(pipeline.device)
    untrained_model.eval()
    
    # 2. Proposed: Trained Resilient Model
    pipeline.fusion_model.eval()
    
    baseline_fused = []
    proposed_fused = []
    batch_size = 64
    
    with torch.no_grad():
        for i in range(0, N, batch_size):
            b_text = text_t[i : i + batch_size]
            b_image = image_dropped_t[i : i + batch_size]
            
            # Project using untrained model
            z_untrained = untrained_model(b_text, b_image)
            baseline_fused.append(z_untrained.cpu().numpy())
            
            # Project using trained dropout-resilient model
            z_trained = pipeline.fusion_model(b_text, b_image)
            proposed_fused.append(z_trained.cpu().numpy())
            
    baseline_fused = np.concatenate(baseline_fused, axis=0)
    proposed_fused = np.concatenate(proposed_fused, axis=0)
    
    # Query database for all unique user graded queries to run evaluation against
    conn = sqlite3.connect(database.DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT query_text, image_path FROM grades")
    queries_rows = cursor.fetchall()
    
    if not queries_rows:
        conn.close()
        return {
            "missing_rate": missing_rate,
            "baseline": {"NDCG@10": 0.0, "MAP": 0.0},
            "proposed": {"NDCG@10": 0.0, "MAP": 0.0},
            "graded_queries_count": 0
        }
        
    baseline_ndcgs = []
    baseline_maps = []
    proposed_ndcgs = []
    proposed_maps = []
    
    for row in queries_rows:
        q_text = row["query_text"]
        q_img = row["image_path"]
        
        # Load human grades for this query
        cursor.execute("SELECT item_id, rating FROM grades WHERE query_text = ? AND image_path = ?", (q_text, q_img))
        grades_dict = {str(r["item_id"]): int(r["rating"]) for r in cursor.fetchall()}
        
        if not grades_dict:
            continue
            
        # Extract query representations
        q_text_emb_t = pipeline.text_model.encode([q_text]).to(pipeline.device) if q_text else None
        q_image_emb_t = None
        if q_img and Path(q_img).exists():
            q_image_emb_t = pipeline.image_model.encode([q_img]).to(pipeline.device)
            
        # Get query fused vectors for baseline and proposed models
        if q_text_emb_t is not None and q_image_emb_t is not None:
            with torch.no_grad():
                q_fused_untrained = untrained_model(q_text_emb_t, q_image_emb_t).cpu().numpy().reshape(-1)
                q_fused_trained = pipeline.fusion_model(q_text_emb_t, q_image_emb_t).cpu().numpy().reshape(-1)
        elif q_text_emb_t is not None:
            zero_img = torch.zeros(1, 2048, device=pipeline.device)
            with torch.no_grad():
                q_fused_untrained = untrained_model(q_text_emb_t, zero_img).cpu().numpy().reshape(-1)
                q_fused_trained = pipeline.fusion_model(q_text_emb_t, zero_img).cpu().numpy().reshape(-1)
        elif q_image_emb_t is not None:
            zero_text = torch.zeros(1, 384, device=pipeline.device)
            with torch.no_grad():
                q_fused_untrained = untrained_model(zero_text, q_image_emb_t).cpu().numpy().reshape(-1)
                q_fused_trained = pipeline.fusion_model(zero_text, q_image_emb_t).cpu().numpy().reshape(-1)
        else:
            continue
            
        # Search baseline
        scores_untrained = np.dot(baseline_fused, q_fused_untrained)
        indices_untrained = np.argsort(scores_untrained)[::-1][:10]
        results_untrained = [str(metadata.iloc[idx]["item_id"]) for idx in indices_untrained]
        
        rels_untrained = [float(grades_dict.get(item_id, 0)) for item_id in results_untrained]
        baseline_ndcgs.append(compute_ndcg_at_k(rels_untrained, k=10))
        baseline_maps.append(compute_ap([grades_dict.get(item_id, 0) >= 3 for item_id in results_untrained]))
        
        # Search proposed
        scores_trained = np.dot(proposed_fused, q_fused_trained)
        indices_trained = np.argsort(scores_trained)[::-1][:10]
        results_trained = [str(metadata.iloc[idx]["item_id"]) for idx in indices_trained]
        
        rels_trained = [float(grades_dict.get(item_id, 0)) for item_id in results_trained]
        proposed_ndcgs.append(compute_ndcg_at_k(rels_trained, k=10))
        proposed_maps.append(compute_ap([grades_dict.get(item_id, 0) >= 3 for item_id in results_trained]))
        
    conn.close()
    
    return {
        "missing_rate": missing_rate,
        "baseline": {
            "NDCG@10": float(np.mean(baseline_ndcgs)) if baseline_ndcgs else 0.0,
            "MAP": float(np.mean(baseline_maps)) if baseline_maps else 0.0
        },
        "proposed": {
            "NDCG@10": float(np.mean(proposed_ndcgs)) if proposed_ndcgs else 0.0,
            "MAP": float(np.mean(proposed_maps)) if proposed_maps else 0.0
        },
        "graded_queries_count": len(baseline_ndcgs)
    }



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
            row["text_similarity"] = r["text_similarity"]
            row["image_similarity"] = r["image_similarity"]
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
