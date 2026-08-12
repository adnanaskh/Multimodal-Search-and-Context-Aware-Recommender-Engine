import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .cv_model import ImageEmbeddingModel
from .nlp_model import TextEmbeddingModel
from .fusion_model import FusionModel


class EmbeddingPipeline:
    def __init__(self, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.text_model = TextEmbeddingModel(device=self.device)
        self.image_model = ImageEmbeddingModel(device=self.device)
        self.fusion_model = FusionModel().to(self.device)

    def build_embeddings(self, metadata_path: str, image_dir: str, output_dir: str) -> None:
        df = pd.read_csv(metadata_path)
        texts = (df["title"] + " " + df["description"].fillna(""))
        texts = texts.astype(str).tolist()
        image_paths = [os.path.join(image_dir, filename) for filename in df["image_filename"].tolist()]

        text_embs = self.text_model.encode(texts)
        image_embs = self.image_model.encode(image_paths)

        fused = []
        batch_size = 32
        for i in range(0, len(text_embs), batch_size):
            batch_text = text_embs[i : i + batch_size].to(self.device)
            batch_image = image_embs[i : i + batch_size].to(self.device)
            with torch.no_grad():
                fused_batch = self.fusion_model(batch_text, batch_image)
            fused.append(fused_batch.cpu())
        fused = torch.cat(fused, dim=0)

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        np.save(Path(output_dir) / "text_embeddings.npy", text_embs.numpy())
        np.save(Path(output_dir) / "image_embeddings.npy", image_embs.numpy())
        np.save(Path(output_dir) / "fused_embeddings.npy", fused.numpy())

        df.to_csv(Path(output_dir) / "metadata.csv", index=False)
        print(f"Saved embeddings and metadata to {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate text, image, and fused embeddings.")
    parser.add_argument("--metadata", default="data/mock_catalog.csv")
    parser.add_argument("--images", default="data/images")
    parser.add_argument("--output", default="data/embeddings")
    args = parser.parse_args()

    pipeline = EmbeddingPipeline()
    pipeline.build_embeddings(args.metadata, args.images, args.output)
