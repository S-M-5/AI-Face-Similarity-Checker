"""
Description:

The primary CLI entry point for running 1-on-1 image comparisons or cross-comparing entire identity directories.

Functions:
    1 - get_embedding(model, img_path, device): Loads an image, detects/aligns the primary face using MTCNN, normalizes the cropped face tensor,
        and passes it through FaceEmbeddingNet to extract its 128-d embedding.

    2 - compute_calibrated_score(emb1, emb2): Calculates raw cosine similarity between two 128-d vectors and maps it onto a calibrated 0% - 100% scale
        based on empirical low/high thresholds (0.20 to 0.85).

    3 - get_valid_image_paths(directory): Scans a folder and returns file paths for valid image formats (.jpg, .png, .webp, .avif).

    4 - show_comparison(img1_path, img2_path, score): Displays a side-by-side Matplotlib visual comparison of two images with color-coded similarity headers.

    5 - main(): Runs the interactive terminal menu offering three operational modes:
            1 - Single 1-on-1 local image comparison.
            2 - Single image vs. scraped identity folder.
            3 - Cross-comparison between two scraped names.
"""


import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from model import FaceEmbeddingNet
from facenet_pytorch import MTCNN

from scrape_faces import build_face_dataset
from clean_dataset import clean_directory

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(image_size=160, margin=10, keep_all=False, select_largest=True, post_process=False, device=device)
def get_embedding(model, img_path, device):
    model.eval()
    if not os.path.exists(img_path):
        return None
        
    try:
        img = Image.open(img_path).convert('RGB')
        
        face_tensor = mtcnn(img)
        
        if face_tensor is None:
            return None
            
        transform_norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        if face_tensor.max() > 1.0:
            face_tensor = face_tensor / 255.0
            
        img_tensor = transform_norm(face_tensor).unsqueeze(0).to(device)
        
        with torch.no_grad():
            embedding = model(img_tensor)
        return embedding

    except Exception as e:
        return None

def compute_calibrated_score(emb1, emb2):

    similarity = F.cosine_similarity(emb1, emb2).item()
    
    baseline_low = 0.20  
    baseline_high = 0.85  

    calibrated = (similarity - baseline_low) / (baseline_high - baseline_low)
    percentage = max(0.0, min(100.0, calibrated * 100))
    return percentage, similarity

def get_valid_image_paths(directory):
    if not os.path.exists(directory):
        return []
    return [
        os.path.join(directory, f) 
        for f in os.listdir(directory) 
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.avif', '.webp'))
    ]

def get_valid_image_paths(directory):
    if not os.path.exists(directory):
        return []
    return [
        os.path.join(directory, f) 
        for f in os.listdir(directory) 
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.avif', '.webp'))
    ]

def show_comparison(img1_path, img2_path, score):
    plt.close('all')
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    fig.canvas.manager.set_window_title('Face AI Matcher')
    
    img1 = Image.open(img1_path).convert('RGB')
    img2 = Image.open(img2_path).convert('RGB')
    
    axes[0].imshow(img1)
    axes[0].axis('off')
    
    axes[1].imshow(img2)
    axes[1].axis('off')
    
    color = '#00b300' if score > 70.0 else '#e60000'
    fig.suptitle(f"Similarity Score: {score:.2f}%", fontsize=18, fontweight='bold', color=color)
    
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)

def main():
    weights_path = "weights/face_similarity_model.pth"
    if not os.path.exists(weights_path):
        print(f"[!] Error: Could not find trained weights at {weights_path}.")
        return

    print("[*] Waking up GPU and loading Face AI Model...")
    model = FaceEmbeddingNet(embedding_dim=128).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()
    print("[+] Model loaded successfully!\n")

    while True:
        print("\n" + "="*50)
        print("Select a mode:")
        print("1. Compare two specific images")
        print("2. Scrape a name and average scores against a target image")
        print("3. Scrape TWO names and cross-compare them")
        print("Type 'quit' or 'q' to exit.")
        choice = input("Choice: ").strip()

        if choice.lower() in ['quit', 'q']: 
            break

        if choice == '1':
            img1_path = input("Enter path to FIRST image: ").strip('\"\'')
            img2_path = input("Enter path to SECOND image: ").strip('\"\'')
            
            emb1 = get_embedding(model, img1_path, device)
            emb2 = get_embedding(model, img2_path, device)
            
            if emb1 is None or emb2 is None:
                print("[!] Error: MTCNN could not detect a valid face in one or both images.")
                continue
                
            percentage, raw_sim = compute_calibrated_score(emb1, emb2)
            show_comparison(img1_path, img2_path, percentage)
            
            print(f"\n---> [*] Raw Cosine Similarity: {raw_sim:.4f}")
            print(f"---> [*] AI Calibrated Score: {percentage:.2f}%")
            print("---> [+] MATCH!\n" if percentage > 70.0 else "---> [-] NO MATCH\n")

        elif choice == '3':
            name_1 = input("Enter FIRST name: ").strip()
            name_2 = input("Enter SECOND name: ").strip()
            num_images = int(input("How many images per person?(Press Enter for 10) ").strip() or "10")
            
            dir_1, _ = build_face_dataset(name_1, max_images=num_images)
            clean_directory(dir_1)
            dir_2, _ = build_face_dataset(name_2, max_images=num_images)
            clean_directory(dir_2)
            
            images_1 = get_valid_image_paths(dir_1)
            images_2 = get_valid_image_paths(dir_2)
            
            scores = []
            best_score = 0
            best_pair = (None, None)
            
            print(f"\n[*] Extracting face embeddings for cross-comparison...")
            
            # Pre-cache embeddings to avoid redundant computation
            embs_1 = [(p, get_embedding(model, p, device)) for p in images_1]
            embs_2 = [(p, get_embedding(model, p, device)) for p in images_2]
            
            # Filter out images where MTCNN detected no face
            embs_1 = [(p, e) for p, e in embs_1 if e is not None]
            embs_2 = [(p, e) for p, e in embs_2 if e is not None]
            
            for path1, e1 in embs_1:
                for path2, e2 in embs_2:
                    pct, raw_sim = compute_calibrated_score(e1, e2)
                    scores.append(pct)
                    if pct > best_score:
                        best_score = pct
                        best_pair = (path1, path2)

            if name_1.lower() == name_2.lower():
                avg_score = 100
            else:
                avg_score = sum(scores)/len(scores)  

            if scores:
                print(f"\n======================================")
                print(f"[*] CROSS-COMPARISON: {name_1.upper()} vs {name_2.upper()}")
                print(f"[*] Valid Face Comparisons: {len(scores)}")
                print(f"[*] AVERAGE SIMILARITY: {avg_score:.2f}%")
                print(f"======================================\n")
                
                if best_pair[0] and best_pair[1]:
                    show_comparison(best_pair[0], best_pair[1], avg_score)
            else:
                print("[-] No valid faces were detected in the downloaded images.")

if __name__ == "__main__":
    main()