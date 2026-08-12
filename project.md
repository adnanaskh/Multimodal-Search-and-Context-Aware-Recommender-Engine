# Honors Project: Multimodal Search and Context-Aware Recommender Engine

## I. Software Requirement Document (SRD)

### 1. Project Title
Multimodal Search and Context-Aware Recommender Engine

### 2. Purpose
To develop an intelligent information retrieval and recommendation system that accepts both text and image inputs (multimodal), processes the semantic meaning of these inputs, and delivers highly personalized search results based on user context and historical interaction data.

### 3. Honors Subjects Integration
* **Computer Vision:** Processing image queries and extracting visual feature embeddings.
* **NLP and Text Analytics:** Parsing user search queries, reviews, and textual item descriptions.
* **Artificial Neural Network:** Utilizing deep learning architectures for multimodal embedding generation.
* **Recommender System:** Providing personalized rankings using hybrid filtering techniques.
* **Data Science and Analytics:** Managing the data pipeline, conducting exploratory data analysis (EDA), and validating search relevance statistically.

### 4. Functional Requirements
* **FR1 (Multimodal Input):** The system must accept text queries, image uploads, or a combination of both.
* **FR2 (Feature Extraction):** The system must extract and project textual and visual features into a shared latent space.
* **FR3 (Personalized Retrieval):** The system must generate recommendations by calculating the similarity between the user's multimodal query and the indexed catalog, weighted by the user's profile.
* **FR4 (Relevance Grading Interface):** The system must include a TryRating-style dashboard for human-in-the-loop data analysts to manually evaluate and grade search query relevance against the retrieved results.

### 5. Non-Functional Requirements
* **NFR1 (Scalability):** The indexing mechanism must efficiently handle large vocabularies and datasets.
* **NFR2 (Latency):** Search queries must return top-k results within 500 milliseconds.
* **NFR3 (Research Viability):** The system must output rigorous evaluation logs for academic benchmarking.

## II. Specification Document (SD)

### 1. Algorithmic Specifications
* **Text Processing (NLP):** Implementation of a transformer-based model (e.g., BERT or RoBERTa) to tokenize and generate dense vector representations of textual data.
* **Image Processing (CV):** Implementation of a Convolutional Neural Network (e.g., ResNet50) or Vision Transformer (ViT) to extract visual feature vectors.
* **Embedding Fusion (ANN):** A Two-Tower Neural Network architecture where the text and image embeddings are concatenated and passed through dense layers to create a unified multimodal vector.
* **Recommendation Engine:** A Hybrid filtering model combining Neural Collaborative Filtering (NCF) for user-item interactions and Content-Based Filtering utilizing the multimodal vectors.

### 2. Indexing Strategy
To ensure the information retrieval component is efficient, the textual metadata will utilize advanced indexing algorithms. We will implement and compare Blocked Sort-Based Indexing (BSBI) and Single-Pass In-Memory Indexing (SPIMI). This comparison will serve as a strong technical contribution to your research paper, demonstrating optimization in handling the inverted index before the deep learning models take over for semantic ranking.

### 3. Mathematical Specifications for Retrieval
Distance between query embeddings and item embeddings will be computed using Cosine Similarity.

`similarity = cos(θ) = (A · B) / (||A|| ||B||)`

### 4. Evaluation Metrics (Data Science & Search Relevance)
The output of the engine will be quantitatively measured using standard search relevance metrics.
* **Normalized Discounted Cumulative Gain (NDCG):** To evaluate the ranking quality of the recommender.
* **Mean Average Precision (MAP):** To evaluate the precision of the multimodal retrieval.

## III. Architecture Document (AD)
The system will follow a microservices-inspired architecture, divided into offline (training/indexing) and online (serving) pipelines.

### 1. System Components

| Module | Description | Technology / Method |
| :--- | :--- | :--- |
| **Client / UI** | User-facing application for submitting text/image queries and interacting with results. | Python(Streamlit) |
| **API Gateway** | Manages routing, user session handling, and payload parsing. | FastAPI / Flask |
| **Embedding Engine** | The ANN component that transforms raw input (CV and NLP) into multimodal vectors. | PyTorch / TensorFlow |
| **Indexing Engine** | Manages the vocabulary and metadata. Utilizes SPIMI for efficient index construction. | Python/C++ |
| **Vector Database** | Stores the generated embeddings and performs Approximate Nearest Neighbor (ANN) search. | FAISS/Milvus |
| **Recommender Core** | Re-ranks the retrieved nearest neighbors based on the user's historical preferences. | Hybrid NCF |
| **Evaluation Hub** | An internal online data analyst portal for sourcing and annotating search relevance data to refine the model. | Custom Dashboard |

### 2. Data Flow (Online Inference Phase)
1. User submits a multimodal query (e.g., an image of a shirt + text "in blue").
2. The API Gateway routes the query to the Embedding Engine.
3. The CV module processes the image; the NLP module processes the text. The ANN fuses them into a single query vector.
4. The query vector is sent to the Vector Database to retrieve the top 100 candidate items.
5. The Recommender Core takes these 100 items, cross-references the user's profile, and re-ranks them.
6. The final top 10 results are returned to the Client.

## IV. Project Structure

The repository follows a package-oriented layout with source code inside `src/` and data assets in `data/`.

- `src/` - Python source package
  - `src/__main__.py` - entrypoint for `python -m src`
  - `src/main.py` - FastAPI service and search/grade endpoints
  - `src/data_loader.py` - dataset cleaning and image resizing
  - `src/nlp_model.py` - text embedding encoder
  - `src/cv_model.py` - image embedding encoder
  - `src/fusion_model.py` - multimodal fusion model
  - `src/embedding_pipeline.py` - generates text, image, and fused embeddings
  - `src/vector_db.py` - FAISS-based similarity search wrapper
  - `src/recommender.py` - personalization and ranking logic
  - `src/utils.py` - utility helpers for path and directory management
- `scripts/` - helper scripts for dataset generation and experimentation
- `tests/` - placeholder and future automated tests
- `data/` - dataset inputs, cleaned outputs, and embeddings

## V. Current Implementation Status

- Day 1: dataset and pipeline setup completed.
- Day 2: multimodal embedding generation and ranking logic implemented.
- Package layout restructured so all source modules are under `src/`.
- Documentation updated to reflect the current file structure and usage commands.
