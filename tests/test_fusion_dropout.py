import os
import torch
import numpy as np
from src.fusion_model import FusionModel
from src.train_fusion import train_fusion_model


def test_fusion_model_dropout():
    # 1. Test dropout in forward pass
    model = FusionModel(text_dim=10, image_dim=20, hidden_dim=16, output_dim=8)
    model.eval()
    
    text_emb = torch.randn(4, 10)
    image_emb = torch.randn(4, 20)
    
    # Run forward pass with 100% dropout on text: text_emb should be zeroed out internally
    out_dropped_text = model(text_emb, image_emb, text_dropout_prob=1.0, image_dropout_prob=0.0)
    
    # Verify that the result with dropped text equals the result with zero text input directly
    out_zero_text = model(torch.zeros_like(text_emb), image_emb, text_dropout_prob=0.0, image_dropout_prob=0.0)
    
    assert torch.allclose(out_dropped_text, out_zero_text, atol=1e-5)



def test_fusion_training_convergence():
    # 2. Test self-supervised contrastive training loop
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Create mock embeddings for 128 products
    mock_text = np.random.randn(128, 384).astype(np.float32)
    mock_image = np.random.randn(128, 2048).astype(np.float32)
    
    # Normalize mock embeddings to mimic encoder outputs
    mock_text = mock_text / np.linalg.norm(mock_text, axis=1, keepdims=True)
    mock_image = mock_image / np.linalg.norm(mock_image, axis=1, keepdims=True)
    
    model = FusionModel(text_dim=384, image_dim=2048, hidden_dim=64, output_dim=32)
    initial_weights = model.fc[0].weight.clone()
    
    # Train for 5 epochs
    trained_model = train_fusion_model(
        text_embs=mock_text,
        image_embs=mock_image,
        model=model,
        epochs=5,
        batch_size=32,
        lr=1e-2,
        consistency_weight=1.0,
        device="cpu"
    )
    
    # Verify weights have changed
    assert not torch.equal(trained_model.fc[0].weight, initial_weights)
    
    # Verify consistency of representations: full multimodal embedding vs unimodal representations
    trained_model.eval()
    t_text = torch.tensor(mock_text[:10])
    t_image = torch.tensor(mock_image[:10])
    
    with torch.no_grad():
        z_full = trained_model(t_text, t_image)
        z_text_only = trained_model(t_text, torch.zeros_like(t_image))
        z_image_only = trained_model(torch.zeros_like(t_text), t_image)
        
    # Check that cosine similarity is reasonably high between views
    for i in range(10):
        sim_text = float(torch.dot(z_full[i], z_text_only[i]))
        sim_image = float(torch.dot(z_full[i], z_image_only[i]))
        
        # In a trained space with consistency regularization, similarity should be positive and high
        assert sim_text > 0.6, f"Text consistency low: {sim_text}"
        assert sim_image > 0.6, f"Image consistency low: {sim_image}"
