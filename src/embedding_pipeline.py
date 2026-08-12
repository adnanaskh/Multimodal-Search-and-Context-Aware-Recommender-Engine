import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .cv_model import ImageEmbeddingModel
from .nlp_model import TextEmbeddingModel
from .fusion_model import FusionModel
from .train_fusion import train_fusion_model


class EmbeddingPipeline:
    def __init__(self, device: str | None = None, checkpoint_path: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.text_model = TextEmbeddingModel(device=self.device)
        self.image_model = ImageEmbeddingModel(device=self.device)
        self.fusion_model = FusionModel().to(self.device)
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            try:
                self.fusion_model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
                print(f"Loaded trained FusionModel from {checkpoint_path}")
            except Exception as e:
                print(f"Could not load FusionModel checkpoint: {e}")
        self.fusion_model.eval()

    def build_embeddings(self, metadata_path: str, image_dir: str, output_dir: str, epochs: int = 15) -> None:
        df = pd.read_csv(metadata_path)
        texts = (df["title"] + " " + df["description"].fillna(""))
        texts = texts.astype(str).tolist()
        image_paths = [os.path.join(image_dir, filename) for filename in df["image_filename"].tolist()]

        text_embs = self.text_model.encode(texts)
        image_embs = self.image_model.encode(image_paths)

        # Train the Fusion Model on the generated text and image representations
        self.fusion_model = train_fusion_model(
            text_embs=text_embs.numpy(),
            image_embs=image_embs.numpy(),
            model=self.fusion_model,
            epochs=epochs,
            batch_size=64,
            lr=1e-3,
            consistency_weight=1.5,
            device=self.device
        )
        
        # Save model checkpoint
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        checkpoint_file = Path(output_dir) / "fusion_model.pt"
        torch.save(self.fusion_model.state_dict(), checkpoint_file)
        print(f"Saved trained FusionModel checkpoint to {checkpoint_file}")

        fused = []
        batch_size = 32
        self.fusion_model.eval()
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

    checkpoint_path = os.path.join(args.output, "fusion_model.pt")
    pipeline = EmbeddingPipeline(checkpoint_path=checkpoint_path)
    pipeline.build_embeddings(args.metadata, args.images, args.output)

