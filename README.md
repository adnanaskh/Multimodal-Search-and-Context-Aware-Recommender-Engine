# Bridging the Modality Gap: Adaptive Feature Dropout and Hybrid Indexing in Multimodal E-Commerce Search

<p align="center">
  <strong>A production-grade multimodal information retrieval system combining Computer Vision, NLP, Neural Networks, and Recommender Systems into a unified, interactive search engine with a real-time evaluation dashboard.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch" alt="PyTorch">
  <img src="https://img.shields.io/badge/FastAPI-0.100+-009688?logo=fastapi" alt="FastAPI">
  <img src="https://img.shields.io/badge/Streamlit-1.x-FF4B4B?logo=streamlit" alt="Streamlit">
  <img src="https://img.shields.io/badge/FAISS-Vector%20Search-764ABC" alt="FAISS">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License">
</p>

---

## One-Command Quick Start

Clone the repository and run a single command to install dependencies, generate the catalog, download product images, train the model, and launch both servers:

```bash
git clone https://github.com/adnanaskh/Multimodal-Search-and-Context-Aware-Recommender-Engine.git
cd Multimodal-Search-and-Context-Aware-Recommender-Engine
python setup_and_run.py
```

Then open **http://localhost:8501** in your browser.

> **Requirements:** Python 3.10+ and pip. The script handles everything else automatically.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Novelty: Adaptive Modality Dropout](#novelty-adaptive-modality-dropout)
- [Honors Subject Integration](#honors-subject-integration)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Manual Setup Guide](#manual-setup-guide)
- [How It Works](#how-it-works)
- [Evaluation Metrics & Dashboard](#evaluation-metrics--dashboard)
- [API Reference](#api-reference)
- [Testing](#testing)
- [License](#license)

---

## Overview

This project implements an end-to-end **Multimodal Search and Personalized Recommender Engine** that:

1. **Accepts text queries, image uploads, or both** — enabling users to search an e-commerce catalog by describing what they want, uploading a photo of a similar product, or combining both for precise retrieval.

2. **Fuses visual and textual features** into a shared 256-dimensional embedding space using a trained Two-Tower neural network with self-supervised contrastive learning.

3. **Provides personalized results** by re-ranking search results based on each user's historical interaction patterns (clicks, purchases, ratings).

4. **Compares two retrieval paradigms** side-by-side — a neural **Semantic Vector Search** (FAISS) against a classical **SPIMI Inverted Index** — with statistical significance testing.

5. **Includes a full evaluation dashboard** with NDCG@10, MAP, latency benchmarks, t-test significance alerts, and a modality ablation simulator.

---

## Key Features

| Feature | Description |
|:---|:---|
| **Multimodal Search** | Text-only, image-only, and combined text+image queries using MiniLM + ResNet50 |
| **Adaptive Modality Dropout** | Novel training technique making the system resilient to missing modalities |
| **Side-by-Side Comparison** | Semantic vector search vs. classical SPIMI inverted index with visual diff |
| **Personalized Ranking** | Context-aware re-ranking based on user interaction history (clicks, purchases, ratings) |
| **Real-Time Evaluation Hub** | NDCG@10, MAP, latency, statistical t-tests, and modality sensitivity charts |
| **Ablation Simulator** | Interactive slider to test system resilience when 0-90% of product modalities are missing |
| **TryRating Interface** | Human-in-the-loop relevance grading (1-5 stars) for search results |
| **200 Unique Products** | Diverse catalog across 10 categories with real product images from Unsplash |
| **Sub-500ms Latency** | Globally cached encoders and FAISS approximate nearest neighbor search |
| **Full API** | RESTful FastAPI backend with 10+ endpoints for search, grading, metrics, and history |

---

## System Architecture

```
┌───────────────────────────────────────────────────────────────┐
│                    Streamlit Frontend (port 8501)              │
│                                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────────┐  │
│  │  Multimodal   │  │   User       │  │  Performance        │  │
│  │  Search Tab   │  │   Activity   │  │  Evaluation Hub     │  │
│  │              │  │   Insights   │  │  (NDCG, MAP, t-Test)│  │
│  └──────┬───────┘  └──────┬───────┘  └──────────┬──────────┘  │
└─────────┼─────────────────┼─────────────────────┼─────────────┘
          │ HTTP             │ HTTP                │ HTTP
          ▼                 ▼                     ▼
┌───────────────────────────────────────────────────────────────┐
│                   FastAPI Backend (port 8000)                  │
│                                                               │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────┐  │
│  │ Embedding  │  │  FAISS     │  │  SPIMI     │  │ SQLite │  │
│  │ Pipeline   │  │  Vector DB │  │  Index     │  │ Logger │  │
│  │            │  │  (3 indices)│  │ (Inverted) │  │        │  │
│  └──────┬─────┘  └─────┬──────┘  └──────┬─────┘  └───┬────┘  │
│         │              │               │             │        │
│  ┌──────┴──────────────┴───────────────┴─────────────┘        │
│  │                                                            │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────────┐             │
│  │  │ MiniLM   │  │ ResNet50 │  │ FusionModel  │             │
│  │  │ (Text)   │  │ (Vision) │  │ (Two-Tower)  │             │
│  │  └──────────┘  └──────────┘  └──────────────┘             │
│  │       384-d         2048-d        → 256-d fused            │
│  └────────────────────────────────────────────────────────────│
└───────────────────────────────────────────────────────────────┘
```

**Three FAISS Indices** are maintained simultaneously:
- **Text-only** index (384-dim MiniLM embeddings)
- **Image-only** index (2048-dim ResNet50 features)
- **Fused** index (256-dim trained multimodal embeddings)

The query routing logic automatically selects the appropriate index based on input modality.

---

## Novelty: Adaptive Modality Dropout

In real-world e-commerce catalogs, products frequently have **missing modalities** — items without images, or with empty descriptions. Traditional fusion models fail catastrophically when a modality is absent (zero-filled vectors collapse the representation).

This project implements **Adaptive Modality Dropout with Multi-View Contrastive Learning** to solve this:

### 1. Structured Modality Dropout

During training, text or image inputs are randomly zeroed out per-sample using Bernoulli masks:

$$H_{fused} = \text{Normalize}\left( \text{FC}\left(\text{cat}\left(X_{text} \odot r_t,\ X_{image} \odot r_v\right)\right) \right)$$

where $r_t \sim \text{Bernoulli}(0.7)$ and $r_v \sim \text{Bernoulli}(0.7)$ are independent retention masks.

### 2. Multi-View Contrastive Loss (InfoNCE)

The training objective aligns three views of each product — full multimodal, text-only, and image-only:

$$\mathcal{L}_{contrastive} = -\log \frac{\exp(\text{sim}(z_i, z_j^+) / \tau)}{\sum_{k=1}^{2N} \exp(\text{sim}(z_i, z_k) / \tau)}$$

### 3. Consistency Regularization

An additional consistency loss directly penalizes divergence between full and partial representations:

$$\mathcal{L}_{cons} = \|H_{fused}^{(text, image)} - H_{fused}^{(text, 0)}\|_2^2 + \|H_{fused}^{(text, image)} - H_{fused}^{(0, image)}\|_2^2$$

### Interactive Ablation Simulator

The dashboard includes an interactive slider that lets you simulate missing-modality rates from 0% to 90%. It benchmarks our dropout-resilient fusion against an untrained concatenation baseline in real-time, demonstrating significant retrieval quality improvements under degraded conditions.

---

## Honors Subject Integration

This project integrates five honors-level computer science subjects:

| Subject | Integration |
|:---|:---|
| **Computer Vision** | ResNet50 extracts 2048-dimensional visual feature vectors from product images. Supports image-based search queries. |
| **NLP & Text Analytics** | MiniLM-L6-v2 (sentence-transformers) generates 384-dimensional semantic text embeddings from product titles and descriptions. |
| **Artificial Neural Networks** | Two-Tower FusionModel projects text + image features into a shared 256-d space using self-supervised contrastive learning with modality dropout. |
| **Recommender Systems** | Hybrid personalization engine re-ranks results using historical user interaction weights (clicks, purchases, ratings) per category. |
| **Data Science & Analytics** | SQLite logging, NDCG@10/MAP evaluation, independent t-test significance testing, and latency benchmarking in a real-time dashboard. |

---

## Technology Stack

| Layer | Technology | Purpose |
|:---|:---|:---|
| **Frontend** | Streamlit | Interactive dashboard with search, grading, and evaluation tabs |
| **Backend API** | FastAPI + Uvicorn | RESTful API with CORS, async startup, and cached model loading |
| **Text Encoder** | HuggingFace MiniLM-L6-v2 | Sentence-level semantic embeddings (384-dim) |
| **Image Encoder** | PyTorch ResNet50 (ImageNet) | Visual feature extraction (2048-dim) |
| **Fusion Network** | PyTorch (custom) | Two-Tower projection with modality dropout (→ 256-dim) |
| **Vector Search** | FAISS (faiss-cpu) | Approximate nearest neighbor search on L2-normalized embeddings |
| **Classical Index** | SPIMI (custom) | Single-Pass In-Memory Indexing for boolean text search |
| **Database** | SQLite | Search history logging, relevance grades, and evaluation data |
| **Statistics** | SciPy | Independent two-sample t-tests for significance evaluation |
| **Image Processing** | Pillow | Image resizing, preprocessing, and format handling |

---

## Project Structure

```
recommender_system/
│
├── setup_and_run.py                  # Single-command setup & launch script
├── requirements.txt                  # Python dependencies
├── README.md                         # This file
├── LICENSE                           # MIT License
├── project.md                        # Software requirements & specification document
├── PROGRESS.md                       # Development progress log
├── research_plan.md                  # Research methodology & novelty analysis
│
├── src/                              # Core application package
│   ├── __init__.py                   # Package init
│   ├── __main__.py                   # Entry point for `python -m src`
│   ├── main.py                       # FastAPI server with all endpoints
│   ├── nlp_model.py                  # Text encoder (MiniLM-L6-v2)
│   ├── cv_model.py                   # Image encoder (ResNet50)
│   ├── fusion_model.py               # Two-Tower FusionModel with modality dropout
│   ├── train_fusion.py               # Self-supervised contrastive training loop
│   ├── embedding_pipeline.py         # End-to-end embedding generation pipeline
│   ├── vector_db.py                  # FAISS vector similarity search wrapper
│   ├── indexer.py                    # SPIMI inverted index builder & boolean search
│   ├── recommender.py                # User profile scoring & personalized re-ranking
│   ├── database.py                   # SQLite schema, logging, and seeding
│   ├── metrics.py                    # NDCG@10, MAP, t-test, and latency metrics
│   ├── data_loader.py                # Image resizing & metadata preprocessing
│   └── utils.py                      # Path and directory utilities
│
├── app.py                            # Streamlit frontend dashboard
├── .streamlit/
│   └── config.toml                   # Streamlit configuration (port, theme)
│
├── scripts/
│   └── generate_unique_catalog.py    # Generates 200 unique products with real images
│
├── tests/                            # Automated test suite
│   ├── test_fusion_dropout.py        # Tests for modality dropout resilience
│   ├── test_indexer.py               # Tests for SPIMI index build and search
│   ├── test_search.py                # Integration tests for API endpoints
│   └── test_placeholder.py           # Smoke test
│
├── data/                             # Data directory (generated at runtime)
│   ├── mock_catalog.csv              # 200 unique product listings
│   ├── user_interactions.csv         # Simulated user click/purchase/rating history
│   ├── images/                       # Product images (downloaded from Unsplash)
│   ├── resized/                      # Preprocessed images (generated)
│   ├── embeddings/                   # Trained model checkpoint & embedding matrices
│   │   ├── text_embeddings.npy       # 200 × 384 text vectors
│   │   ├── image_embeddings.npy      # 200 × 2048 image vectors
│   │   ├── fused_embeddings.npy      # 200 × 256 trained fused vectors
│   │   ├── fusion_model.pt           # Trained FusionModel checkpoint
│   │   └── metadata.csv             # Synced catalog metadata
│   └── query_images/                 # Uploaded query images (runtime)
│
└── .github/
    └── workflows/
        └── ci.yml                    # GitHub Actions CI pipeline
```

---

## Manual Setup Guide

If you prefer to run each step individually:

### 1. Clone & Install

```bash
git clone https://github.com/adnanaskh/Multimodal-Search-and-Context-Aware-Recommender-Engine.git
cd Multimodal-Search-and-Context-Aware-Recommender-Engine
pip install -r requirements.txt
```

### 2. Generate Product Catalog & Download Images

```bash
python scripts/generate_unique_catalog.py
```

This generates:
- `data/mock_catalog.csv` — 200 unique products across 10 categories
- `data/images/*.jpg` — Real product images from Unsplash
- `data/user_interactions.csv` — Simulated interactions for 15 users

### 3. Preprocess Images

```bash
python src/data_loader.py --csv data/mock_catalog.csv --images data/images --output data/resized
```

### 4. Train FusionModel & Generate Embeddings

```bash
python -m src.embedding_pipeline --metadata data/mock_catalog.csv --images data/images --output data/embeddings
```

This takes ~2 minutes on CPU and produces:
- Text embeddings (200 × 384)
- Image embeddings (200 × 2048)
- Trained FusionModel checkpoint
- Fused embeddings (200 × 256)

### 5. Start Backend Server

```bash
python -m src
```

FastAPI starts on **http://localhost:8000**

### 6. Start Frontend Dashboard

In a separate terminal:

```bash
streamlit run app.py
```

Streamlit starts on **http://localhost:8501**

---

## How It Works

### Search Pipeline

```
User Query (text / image / both)
        │
        ▼
┌─────────────────┐
│ Feature          │
│ Extraction       │
│ MiniLM + ResNet50│
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌─────────────────┐
│ Semantic Search  │     │ Classical Search │
│ FAISS Vector DB  │     │ SPIMI Inverted   │
│ (cosine sim)     │     │ Index (boolean)  │
└────────┬────────┘     └────────┬────────┘
         │                       │
         ▼                       ▼
┌─────────────────┐     ┌─────────────────┐
│ Personalized     │     │ TF-IDF Ranked   │
│ Re-Ranking       │     │ Results         │
│ (user profile)   │     │                 │
└────────┬────────┘     └────────┬────────┘
         │                       │
         └───────────┬───────────┘
                     ▼
         ┌─────────────────┐
         │ Side-by-Side     │
         │ Display + Grading│
         └─────────────────┘
```

### Embedding Fusion Architecture

```
Text Input                          Image Input
    │                                   │
    ▼                                   ▼
┌──────────┐                    ┌──────────────┐
│ MiniLM   │                    │ ResNet50     │
│ Encoder  │                    │ (no FC layer)│
└────┬─────┘                    └──────┬───────┘
     │ 384-d                           │ 2048-d
     │                                 │
     │    ┌──── Modality Dropout ────┐  │
     │    │  r_t ~ Bernoulli(0.7)   │  │
     │    │  r_v ~ Bernoulli(0.7)   │  │
     │    └─────────────────────────┘  │
     │                                 │
     └──────────┬──────────────────────┘
                │ concat(384 + 2048 = 2432)
                ▼
        ┌──────────────┐
        │ FC(2432, 512) │
        │ ReLU + BN     │
        │ FC(512, 256)  │
        │ L2 Normalize  │
        └──────┬───────┘
               │ 256-d fused
               ▼
         FAISS Index
```

---

## Evaluation Metrics & Dashboard

The **Performance Evaluation Hub** in the Streamlit dashboard provides five publication-ready evaluation results:

| Metric | Description |
|:---|:---|
| **Search Latency** | Average query time (ms) comparing FAISS vector search vs. SPIMI boolean search |
| **NDCG@10** | Normalized Discounted Cumulative Gain measuring ranking quality based on human ratings |
| **MAP** | Mean Average Precision for relevant results (rated ≥ 3 stars) |
| **t-Test Significance** | Independent two-sample t-test on NDCG/MAP distributions (p < 0.05 threshold) |
| **Modality Sensitivity** | Relative influence weights of text vs. image features on search output |

### Ablation Study Simulator

The interactive slider lets you simulate catalog degradation by randomly zeroing out text or image modalities for 0% to 90% of products. The dashboard benchmarks:
- **Trained dropout-resilient fusion** (our model) vs.
- **Untrained concatenation baseline**

in real-time, demonstrating the practical impact of the adaptive modality dropout novelty.

---

## API Reference

The FastAPI backend exposes the following endpoints on `http://localhost:8000`:

| Method | Endpoint | Description |
|:---|:---|:---|
| `POST` | `/search` | Multimodal search with text query and/or image upload |
| `POST` | `/grade` | Submit a relevance rating (1-5 stars) for a search result |
| `GET` | `/metrics` | Retrieve system evaluation metrics (NDCG, MAP, latency, t-test) |
| `GET` | `/user-history/{user_id}` | Get a user's interaction history and category preferences |
| `GET` | `/search-history` | Retrieve all logged search queries and their results |
| `GET` | `/search-results/{search_id}` | Get detailed results for a specific search query |
| `GET` | `/docs` | Interactive Swagger API documentation |

### Example Search Request

```bash
curl -X POST http://localhost:8000/search \
  -F "query=wireless headphones noise cancellation" \
  -F "user_id=u001" \
  -F "top_k=10"
```

---

## Testing

Run the automated test suite:

```bash
pip install pytest
python -m pytest -v
```

**Test coverage includes:**
- `test_fusion_dropout.py` — Verifies modality dropout resilience (text-only, image-only, full input)
- `test_indexer.py` — SPIMI index construction and boolean search correctness
- `test_search.py` — End-to-end API integration tests (search, grading, metrics)
- `test_placeholder.py` — Basic smoke test

### CI/CD

GitHub Actions CI runs on every push and PR against `main`, testing across Python 3.10, 3.11, and 3.12.

---

## Product Catalog

The system ships with a **200-product catalog** of real consumer products across 10 categories, each with:

- **Unique title** — No two products share the same name
- **Detailed description** — Realistic feature-rich descriptions (not templates)
- **Real product images** — High-quality photos from Unsplash
- **Proper category alignment** — Products are categorized correctly
- **Realistic pricing** — Market-appropriate price points

| Category | Products | Examples |
|:---|:---:|:---|
| Electronics | 20 | Sony WH-1000XM5, Canon EOS R50, DJI Mini 4 Pro |
| Clothing | 20 | Patagonia Better Sweater, Levi's 501 Jeans, Adidas Ultraboost 23 |
| Home | 20 | Dyson V15 Detect, KitchenAid Stand Mixer, Le Creuset Dutch Oven |
| Fitness | 20 | Garmin Forerunner 265, TRX Suspension Trainer, Theragun Elite |
| Beauty | 20 | Dyson Airwrap, SK-II Treatment Essence, Tom Ford Black Orchid |
| Sports | 20 | Wilson Evolution Basketball, Babolat Pure Aero, Burton Custom Snowboard |
| Outdoors | 20 | MSR Hubba Hubba Tent, Leatherman Wave+, Garmin inReach Mini 2 |
| Office | 20 | MacBook Air M3, Keychron Q1 Pro, Herman Miller Aeron (replica) |
| Accessories | 20 | Apple Watch Series 9, Peak Design Sling, Oakley Holbrook |
| Toys | 20 | LEGO Technic Lamborghini, Nintendo Switch OLED, Catan Board Game |

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  Built with PyTorch, FastAPI, Streamlit, and FAISS<br>
  <strong>Bridging the Modality Gap: Adaptive Feature Dropout and Hybrid Indexing in Multimodal E-Commerce Search</strong>
</p>
