"""
Description:

Handles model training using Triplet Margin Loss, Automatic Mixed Precision (AMP), and dynamic loss plotting.

Main Function: train
    - Configures GPU hardware acceleration (AMD ROCm / CUDA).
    - Loads TripletFaceDataset into a multi-worker DataLoader.
    - Concatenates Anchor, Positive, and Negative images into single forward passes to optimize batch normalization statistics.
    - Computes TripletMarginLoss under torch.amp.autocast("cuda").
    - Optimizes un-frozen weights via AdamW and steps a CosineAnnealingLR scheduler.
    - Renders a real-time interactive Matplotlib loss curve during execution and exports model weights to weights/face_similarity_model.pth.
"""


import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import matplotlib.pyplot as plt

from model import FaceEmbeddingNet
from dataset import TripletFaceDataset, train_transforms

def train(data_dir="data", epochs=15, batch_size=128, learning_rate=0.0003, margin=0.3):
    print(f"[*] Starting Fine-Tuning Pipeline...")
    
    # 1. Hardware Setup (Utilize your RX 7800 XT)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Using device: {device}")

    # 2. Load Dataset
    print(f"[*] Loading dataset from '{data_dir}'...")
    try:
        # This will automatically pair Anchors, Positives, and Negatives
        dataset = TripletFaceDataset(root_dir=data_dir, transform=train_transforms)
        # Bumping num_workers to 8 to utilize your Ryzen CPU cores!
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
        print(f"[+] Found {len(dataset.classes)} unique people. Total triplets per epoch: {len(dataset)}")
    except Exception as e:
        print(f"[!] Dataset Error: {e}")
        print("Make sure you have scraped at least 2 people, and each person has 2+ images.")
        return

    model = FaceEmbeddingNet(embedding_dim=128, freeze_backbone=True).to(device)
    triplet_loss = nn.TripletMarginLoss(margin=margin, p=2)

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay=1e-4)

    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler("cuda")

    print("[*] Initializing dynamic training graph...")
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 6))
    
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
    except:
        plt.style.use('ggplot')
        
    epoch_history = []
    loss_history = []

    print("\n[*] Beginning Training Loop...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}/{epochs}")
        for batch_idx, (anchor_img, pos_img, neg_img) in pbar:
            anchor_img = anchor_img.to(device)
            pos_img = pos_img.to(device)
            neg_img = neg_img.to(device)
            
            optimizer.zero_grad()
            
            with autocast("cuda"):
                batch_size_current = anchor_img.size(0)
    
                all_imgs = torch.cat([anchor_img, pos_img, neg_img], dim=0)
        
                all_embs = model(all_imgs)
                
                emb_a, emb_p, emb_n = torch.split(all_embs, batch_size_current)
                
                loss = triplet_loss(emb_a, emb_p, emb_n)

    
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
            
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
            
        scheduler.step()
        
        avg_loss = running_loss / len(dataloader)
        print(f"[+] Epoch {epoch+1} Completed | Average Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}\n")

        epoch_history.append(epoch + 1)
        loss_history.append(avg_loss)
        
        ax.clear()
        
        ax.plot(epoch_history, loss_history, color='#00ffcc', marker='o', linewidth=2.5, markersize=8, label='Triplet Loss')
        
        ax.set_title("Face AI - Training Loss Optimization", fontsize=16, fontweight='bold', color='#333')
        ax.set_xlabel("Epoch", fontsize=12, fontweight='bold')
        ax.set_ylabel("Average Loss", fontsize=12, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, linestyle='--', alpha=0.7)
        
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.1)

    os.makedirs("weights", exist_ok=True)
    save_path = "weights/face_similarity_model.pth"
    torch.save(model.state_dict(), save_path)
    print(f"[*] Training Complete! Model weights saved to '{save_path}'")
    
    plt.ioff() 
    graph_path = "weights/training_loss_graph.png"
    plt.savefig(graph_path, dpi=300, bbox_inches='tight')
    print(f"[*] Graph saved to '{graph_path}'")
    
    print("\n[+] Close the graph window to finish the script.")
    plt.show()
    
if __name__ == "__main__":
    train(data_dir="data", epochs=15, batch_size=128, learning_rate=3e-5, margin=0.3)