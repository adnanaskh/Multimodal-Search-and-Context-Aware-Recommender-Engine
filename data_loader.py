import os
from pathlib import Path
from typing import List

import pandas as pd
from PIL import Image


def load_text_data(csv_path: str) -> pd.DataFrame:
    """Load tabular metadata into a DataFrame."""
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["item_id", "title", "description"])
    df["item_id"] = df["item_id"].astype(str)
    df["title"] = df["title"].astype(str).str.strip()
    df["description"] = df["description"].astype(str).str.strip()
    return df


def clean_text(text: str) -> str:
    """Basic text cleanup for metadata."""
    return (
        text.lower()
        .replace("\n", " ")
        .replace("\r", " ")
        .strip()
    )


def resize_image(image_path: str, output_dir: str, size=(224, 224)) -> str:
    """Resize a product image and save it to the output directory."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    image = Image.open(image_path).convert("RGB")
    image = image.resize(size)
    target_path = Path(output_dir) / Path(image_path).name
    image.save(target_path)
    return str(target_path)


def build_dataset(csv_path: str, image_dir: str, output_dir: str) -> pd.DataFrame:
    """Load metadata, validate images, and create a cleaned dataset."""
    df = load_text_data(csv_path)
    df["clean_title"] = df["title"].apply(clean_text)
    df["clean_description"] = df["description"].apply(clean_text)
    df["image_path"] = df["image_filename"].apply(
        lambda fname: os.path.join(image_dir, fname)
    )
    df["resized_image_path"] = df["image_path"].apply(
        lambda path: resize_image(path, output_dir)
    )
    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare a multimodal dataset.")
    parser.add_argument("--csv", default="data/mock_catalog.csv", help="Path to item metadata CSV.")
    parser.add_argument("--images", default="data/images", help="Path to product images.")
    parser.add_argument("--output", default="data/resized", help="Output directory for resized images.")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        raise FileNotFoundError(f"CSV file not found: {args.csv}")

    dataset = build_dataset(args.csv, args.images, args.output)
    dataset.to_csv("data/cleaned_dataset.csv", index=False)
    print("Saved cleaned dataset to data/cleaned_dataset.csv")
