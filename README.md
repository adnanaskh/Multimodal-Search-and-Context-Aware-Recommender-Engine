# Multimodal Search and Context-Aware Recommender Engine

## Overview

This repository implements a Python-based multimodal search and recommendation engine using text and image embeddings, FAISS vector search, and personalized ranking.

## Repository Structure

- `src/` - main source package for the recommender pipeline and API.
  - `src/data_loader.py` - metadata cleaning, image resizing, and dataset preparation.
  - `src/nlp_model.py` - text embedding model using HuggingFace transformers.
  - `src/cv_model.py` - image embedding model using a pretrained ResNet50.
  - `src/fusion_model.py` - multimodal fusion network for unified embeddings.
  - `src/embedding_pipeline.py` - build text/image/fused embeddings from catalog data.
  - `src/vector_db.py` - FAISS vector index wrapper for similarity search.
  - `src/recommender.py` - personalization and candidate re-ranking logic.
  - `src/main.py` - FastAPI service for search and grading endpoints.
  - `src/__main__.py` - package entrypoint for running the API with `python -m src`.
- `data/` - dataset assets, generated catalogs, and embedding outputs.
- `scripts/` - helper scripts such as mock data generation.
- `tests/` - unit test placeholders and future automated tests.
- `requirements.txt` - project dependencies.
- `project.md` - functional and architectural design document.
- `PROGRESS.md` - implementation timeline and status tracker.

## Setup

1. Create a Python virtual environment and activate it.
   - Windows PowerShell: `python -m venv .venv` then `.\.venv\Scripts\Activate.ps1`
   - Windows CMD: `python -m venv .venv` then `.\.venv\Scripts\activate.bat`
2. Install dependencies:
   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

## Generate Mock Data

Use the helper script in `scripts/` to create a catalog and product images:

```bash
python scripts/generate_mock_data.py
```

This produces:
- `data/mock_catalog.csv`
- `data/images/*.jpg`
- `data/user_interactions.csv`

## Data Preparation

Resize and clean item metadata:

```bash
python src/data_loader.py --csv data/mock_catalog.csv --images data/images --output data/resized
```

The cleaned dataset is saved as `data/cleaned_dataset.csv`.

## Build Embeddings

Generate text, image, and fused embeddings:

```bash
python -m src.embedding_pipeline --metadata data/mock_catalog.csv --images data/images --output data/embeddings
```

This writes:
- `data/embeddings/text_embeddings.npy`
- `data/embeddings/image_embeddings.npy`
- `data/embeddings/fused_embeddings.npy`
- `data/embeddings/metadata.csv`

## Run API Server

Start the FastAPI app from the repository root:

```bash
python -m src
```

Or with Uvicorn directly:

```bash
uvicorn src.main:app --reload
```

### API Endpoints

- `POST /search` - search with `user_id` and `query` form data.
- `POST /grade` - log manual relevance grades with `user_id`, `item_id`, and `rating`.

## Next Steps

1. Implement classical text indexing and baseline boolean search.
2. Extend the frontend with a Streamlit dashboard for query entry and relevance grading.
3. Add evaluation metrics for NDCG and MAP.
4. Add more robust unit tests in `tests/`.
