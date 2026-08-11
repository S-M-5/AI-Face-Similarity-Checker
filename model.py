"""
Description:

Defines the deep learning network that extracts 128-dimensional facial embedding vectors from input images.

Class: FaceEmbeddingNet(nn.Module)
    Role: A fine-tuned ResNet18 feature extractor that maps face images into a 128-dimensional metric space.
    Methods:
        1 - __init__(self, embedding_dim=128, freeze_backbone=True): Loads pre-trained ResNet18 weights, freezes early layers (layer1 - layer3),
            unfreezes layer4, and replaces the default classification head with a custom bottleneck projection block 
            (Linear -> BatchNorm1d -> ReLU -> Dropout -> Linear).

        2 - forward(self, x): Passes normalized face images through the backbone and applies L2 normalization 
            to project outputs onto a hypersphere for cosine similarity calculations.
"""


import torch
import torch.nn as nn
import torchvision.models as models

class FaceEmbeddingNet(nn.Module):
    def __init__(self, embedding_dim=128, freeze_backbone=True):
        super(FaceEmbeddingNet, self).__init__()
        
        # ResNet18 is used for its speed, capability, and memory efficiency
        weights = models.ResNet18_Weights.DEFAULT
        self.backbone = models.resnet18(weights=weights)
        

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
                
            for param in self.backbone.layer4.parameters():
                param.requires_grad = True

                
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 512, bias=True),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, embedding_dim, bias=True)
        )
        
    def forward(self, x):
        x = self.backbone(x)
        x = torch.nn.functional.normalize(x, p=2, dim=1)
        
        return x

# Test block 
if __name__ == "__main__":
    dummy_images = torch.randn(2, 3, 160, 160) 
    
    model = FaceEmbeddingNet(embedding_dim=128)
    
    output_embeddings = model(dummy_images)
    
    print("Model initialized successfully!")
    print(f"Input shape:  {dummy_images.shape}")
    print(f"Output shape: {output_embeddings.shape}") 
    # Expected Output shape: [2, 128]