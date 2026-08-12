import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple


class EmbeddingDataset(Dataset):
    def __init__(self, text_embs: np.ndarray, image_embs: np.ndarray):
        self.text_embs = torch.tensor(text_embs, dtype=torch.float32)
        self.image_embs = torch.tensor(image_embs, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.text_embs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.text_embs[idx], self.image_embs[idx]


def info_nce_loss(features_a: torch.Tensor, features_b: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """Computes NT-Xent / InfoNCE loss between two views of items."""
    N = features_a.size(0)
    device = features_a.device
    
    # Compute cosine similarity matrix
    similarity_matrix = torch.matmul(features_a, features_b.T) / temperature
    
    # The targets are the diagonal elements (corresponding pairs)
    labels = torch.arange(N, device=device)
    loss = nn.CrossEntropyLoss()(similarity_matrix, labels)
    return loss


def train_fusion_model(
    text_embs: np.ndarray,
    image_embs: np.ndarray,
    model: nn.Module,
    epochs: int = 15,
    batch_size: int = 64,
    lr: float = 1e-3,
    consistency_weight: float = 1.5,
    device: str = "cpu",
) -> nn.Module:
    """Trains the FusionModel using structured modality dropout, contrastive loss,

    and consistency loss.
    """
    model = model.to(device)
    dataset = EmbeddingDataset(text_embs, image_embs)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    mse_loss = nn.MSELoss()
    
    print(f"Starting FusionModel training for {epochs} epochs on {device}...")
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_contrastive = 0.0
        total_consistency = 0.0
        
        for text_batch, image_batch in dataloader:
            text_batch = text_batch.to(device)
            image_batch = image_batch.to(device)
            
            optimizer.zero_grad()
            
            # 1. Forward pass with structured modality dropout (active during training)
            z_dropout = model(text_batch, image_batch, text_dropout_prob=0.3, image_dropout_prob=0.3)
            
            # 2. Forward pass without structured modality dropout (full view)
            z_full = model(text_batch, image_batch, text_dropout_prob=0.0, image_dropout_prob=0.0)
            
            # 3. Forward pass with single-modality representations (simulating missing text or image)
            z_text = model(text_batch, torch.zeros_like(image_batch), text_dropout_prob=0.0, image_dropout_prob=0.0)
            z_image = model(torch.zeros_like(text_batch), image_batch, text_dropout_prob=0.0, image_dropout_prob=0.0)
            
            # 4. Consistency loss (MSE)
            loss_cons = mse_loss(z_full, z_text) + mse_loss(z_full, z_image)
            
            # 5. Multi-View Contrastive Loss (InfoNCE)
            loss_cont_1 = info_nce_loss(z_full, z_text)
            loss_cont_2 = info_nce_loss(z_full, z_image)
            loss_cont_3 = info_nce_loss(z_text, z_image)
            loss_cont_dropout = info_nce_loss(z_full, z_dropout)
            
            loss_contrastive = (loss_cont_1 + loss_cont_2 + loss_cont_3 + loss_cont_dropout) / 4.0
            
            # 6. Combined loss
            loss = loss_contrastive + consistency_weight * loss_cons
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_contrastive += loss_contrastive.item()
            total_consistency += loss_cons.item()
            
        avg_loss = total_loss / len(dataloader)
        avg_contrastive = total_contrastive / len(dataloader)
        avg_consistency = total_consistency / len(dataloader)
        
        if epoch % 5 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:02d}/{epochs:02d} | "
                f"Loss: {avg_loss:.4f} (Contrastive: {avg_contrastive:.4f}, Consistency: {avg_consistency:.4f})"
            )
            
    print("FusionModel training complete!")
    return model
