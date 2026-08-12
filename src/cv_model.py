from typing import List

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision import models


class ImageEmbeddingModel:
    def __init__(self, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.resnet50(pretrained=True)
        self.model.fc = torch.nn.Identity()
        self.model = self.model.to(self.device)
        self.model.eval()
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def encode(self, image_paths: List[str], batch_size: int = 16) -> torch.Tensor:
        all_embeddings = []
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i : i + batch_size]
            images = [self.transform(Image.open(path).convert("RGB")) for path in batch_paths]
            inputs = torch.stack(images).to(self.device)
            with torch.no_grad():
                embeddings = self.model(inputs)
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings.cpu())
        return torch.cat(all_embeddings, dim=0)
