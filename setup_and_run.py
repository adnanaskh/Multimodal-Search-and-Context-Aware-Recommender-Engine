"""
setup_and_run.py - One-command setup and launch script.

Usage:
    python setup_and_run.py

This script handles the entire project lifecycle:
  1. Installs Python dependencies
  2. Generates the unique product catalog (200 products)
  3. Downloads real product images from Unsplash
  4. Preprocesses and resizes images
  5. Trains the FusionModel and generates embeddings
  6. Starts the FastAPI backend and Streamlit frontend
"""
import subprocess
import sys
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

def run(cmd, desc, cwd=None, check=True):
    """Run a command with a description banner."""
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"{'='*60}\n")
    result = subprocess.run(
        cmd, shell=True, cwd=cwd or BASE_DIR,
        check=check, text=True
    )
    return result.returncode == 0


def main():
    print(r"""
    ___  ___      _ _   _                     _       _
    |  \/  |     | | | (_)                   | |     | |
    | .  . |_   _| | |_ _ _ __ ___   ___   __| | __ _| |
    | |\/| | | | | | __| | '_ ` _ \ / _ \ / _` |/ _` | |
    | |  | | |_| | | |_| | | | | | | (_) | (_| | (_| | |
    \_|  |_/\__,_|_|\__|_|_| |_| |_|\___/ \__,_|\__,_|_|

    Bridging the Modality Gap: Adaptive Feature Dropout and Hybrid Indexing in Multimodal E-Commerce Search
    ========================================================================================================
    """)

    # Step 1: Install dependencies
    run(
        f"{sys.executable} -m pip install -r requirements.txt -q",
        "Step 1/5: Installing dependencies..."
    )

    # Step 2: Generate catalog + download images + create interactions
    catalog_path = BASE_DIR / "data" / "mock_catalog.csv"
    images_dir = BASE_DIR / "data" / "images"
    needs_catalog = not catalog_path.exists() or len(list(images_dir.glob("*.jpg"))) < 100

    if needs_catalog:
        run(
            f"{sys.executable} scripts/generate_unique_catalog.py",
            "Step 2/5: Generating 200 unique products & downloading images..."
        )
    else:
        print(f"\n{'='*60}")
        print(f"  Step 2/5: Catalog already exists ({len(list(images_dir.glob('*.jpg')))} images) - skipping")
        print(f"{'='*60}")

    # Step 3: Preprocess and resize images
    run(
        f"{sys.executable} src/data_loader.py --csv data/mock_catalog.csv --images data/images --output data/resized",
        "Step 3/5: Preprocessing & resizing product images..."
    )

    # Step 4: Train FusionModel and generate embeddings
    embeddings_dir = BASE_DIR / "data" / "embeddings"
    needs_training = not (embeddings_dir / "fusion_model.pt").exists()

    if needs_training:
        run(
            f"{sys.executable} -m src.embedding_pipeline --metadata data/mock_catalog.csv --images data/images --output data/embeddings",
            "Step 4/5: Training FusionModel & generating embeddings (~2 min on CPU)..."
        )
    else:
        print(f"\n{'='*60}")
        print(f"  Step 4/5: Trained model already exists - skipping")
        print(f"{'='*60}")

    # Step 5: Start both servers
    print(f"\n{'='*60}")
    print(f"  Step 5/5: Launching servers...")
    print(f"{'='*60}\n")

    # Start FastAPI backend
    backend = subprocess.Popen(
        [sys.executable, "-m", "src"],
        cwd=BASE_DIR
    )
    print("  [*] FastAPI backend starting on http://localhost:8000")

    # Wait a moment for the backend to initialize
    import time
    time.sleep(5)

    # Start Streamlit frontend
    frontend = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "app.py"],
        cwd=BASE_DIR
    )
    print("  [*] Streamlit frontend starting on http://localhost:8501")

    print(f"\n{'='*60}")
    print("  ALL SYSTEMS RUNNING!")
    print("  Open http://localhost:8501 in your browser")
    print("  Press Ctrl+C to stop all servers")
    print(f"{'='*60}\n")

    try:
        backend.wait()
    except KeyboardInterrupt:
        print("\nShutting down...")
        backend.terminate()
        frontend.terminate()
        backend.wait()
        frontend.wait()
        print("All servers stopped.")


if __name__ == "__main__":
    main()
