# Multimodal Search and Context-Aware Recommender Engine

This repository implements a production-grade, 100% Python-based Multimodal Search and Personalized Recommender Engine. It combines visual features (ResNet50) and textual metadata semantics (MiniLM) into a unified vector space, integrates a classical SPIMI inverted index for baseline comparison, and provides a real-time significance testing dashboard (TryRating style) evaluating NDCG@10 and MAP.

---

## 🎯 System Architecture

```
   ┌──────────────────────────────────────────────────────────┐
   │                  Streamlit Frontend UI                   │
   │   - Side-by-Side Search Results (Semantic vs Classical)  │
   │   - User Activity Insights & Explainable AI badges       │
   │   - TryRating Relevance Grading (Star feedbacks)         │
   │   - Academic Publication Benchmarks & t-Test Alerts     │
   └─────────────┬──────────────────────────────▲─────────────┘
                 │ (HTTP POST /search, /grade)   │ (HTTP GET /metrics)
                 ▼                              │
   ┌────────────────────────────────────────────┴─────────────┐
   │                   FastAPI Backend API                    │
   │   - Globally Cached Encoders (Sub-500ms Latency)         │
   │   - Multi-Index Vector DB & SPIMI Inverted Index         │
   │   - SQLite Logging (Query parameters, latency, results)  │
   └──────────────────────────────────────────────────────────┘
```

---

## 🛠️ Repository Directory Structure

*   `src/` - Core application package:
    *   `src/database.py` - SQLite schema initializing search history logs and manual grades.
    *   `src/metrics.py` - Relevance evaluation engine computing NDCG@10, Mean Average Precision (MAP), latency, statistical independent t-tests, and modality sensitivity contributions.
    *   `src/main.py` - FastAPI entrypoint caching neural networks and exposing search/grading endpoints.
    *   `src/recommender.py` - Scoring algorithm re-ranking vector results based on historical category engagement weights.
    *   `src/indexer.py` - Single-Pass In-Memory Indexer (SPIMI) build and boolean search engine.
    *   `src/nlp_model.py` - Text feature extractor using Hugging Face `MiniLM-L6-v2`.
    *   `src/cv_model.py` - Image visual feature extractor using pre-trained PyTorch `ResNet50`.
    *   `src/fusion_model.py` - Two-Tower projection network mapping cross-modal towers to a 256-dim space.
    *   `src/embedding_pipeline.py` - Catalog embedding generation pipeline.
    *   `src/data_loader.py` - Preprocessing script (text cleaning and image resizing).
*   `app.py` - Streamlit dashboard with side-by-side search columns and statistical evaluation metrics.
*   `scripts/generate_mock_data.py` - Generates a mock catalog of 3,000 products and interaction history for 10 users.
*   `tests/` - Automated unit/integration tests.

---

## 🚀 Quick Start Guide

### 1. Set Up Virtual Environment & Dependencies
Create a Python virtual environment and install requirements:
```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # Windows PowerShell
pip install -r requirements.txt
```

### 2. Generate 3,000 Product Catalog & Interactions
Generate the e-commerce product listings (titles, visual assets, prices) and user interactions logs:
```bash
python scripts/generate_mock_data.py
```
This produces:
*   `data/mock_catalog.csv` (3,000 rows)
*   `data/images/*.jpg` (3,000 product image files)
*   `data/user_interactions.csv` (mock clicks, views, ratings, purchases for 10 users)

### 3. Run Preprocessing Loader
Resize the 3,000 product images and format metadata fields:
```bash
python src/data_loader.py --csv data/mock_catalog.csv --images data/images --output data/resized
```

### 4. Build Product Embeddings
Generate semantic vector representations for all catalog listings (takes about 1-2 minutes on CPU):
```bash
python -m src.embedding_pipeline --metadata data/mock_catalog.csv --images data/images --output data/embeddings
```
This writes:
*   `data/embeddings/text_embeddings.npy` (3,000 vectors of size 384)
*   `data/embeddings/image_embeddings.npy` (3,000 vectors of size 2048)
*   `data/embeddings/fused_embeddings.npy` (3,000 vectors of size 256)
*   `data/embeddings/metadata.csv` (synced catalog details)

### 5. Start Backend API Server
Start the FastAPI server (caching models at startup for rapid search execution):
```bash
python -m src
```

### 6. Start Streamlit Frontend
In a separate terminal, launch the dashboard:
```bash
streamlit run app.py
```
Streamlit will read from `.streamlit/config.toml` to run headlessly on port `8501`. Navigate to `http://localhost:8501`.

---

## 🔬 Research Publication Metrics (TryRating Dashboard)

The **Analyst Evaluation Hub** compiles 5 publication-ready results based on grading logs:

1.  **Average Search Latency**: Measures query time in milliseconds, comparing optimized FAISS VSD (vector search) with SPIMI index search.
2.  **Ranking Quality (NDCG@10)**: Evaluates ranking effectiveness based on logarithmic discounting of human ratings (1 to 5 stars).
3.  **Retrieval Precision (MAP)**: Measures mean average precision for relevant search results (items rated $\ge 3$).
4.  **Empirical t-Test Significance**: Runs an independent two-sample t-test on NDCG and MAP distributions, rendering a green success banner if the performance gap between vector and classical models is statistically significant ($p < 0.05$).
5.  **Modality Gating Sensitivity**: Computes the average relative influence weight of textual terms vs visual designs on multimodal search outputs.
