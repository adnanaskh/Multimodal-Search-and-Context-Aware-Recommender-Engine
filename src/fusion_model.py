import torch
import torch.nn as nn


class FusionModel(nn.Module):
    def __init__(self, text_dim: int = 384, image_dim: int = 2048, hidden_dim: int = 512, output_dim: int = 256):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(text_dim + image_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, text_emb: torch.Tensor, image_emb: torch.Tensor) -> torch.Tensor:
        x = torch.cat([text_emb, image_emb], dim=1)
        x = self.fc(x)
        return torch.nn.functional.normalize(x, p=2, dim=1)
