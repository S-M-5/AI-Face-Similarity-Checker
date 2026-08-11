"""
Description:

Sanitizes downloaded image folders to clean up multi-face clutter and non-face noise before training.

Functions:
    1 - is_clean_image(image_path): Runs MTCNN detection to verify that an image contains exactly one face 
        (filtering out landscapes, corrupted downloads, and group shots).
    
    2 - clean_directory(dataset_dir): Iterates through an identity folder,
        deletes non-clean images, and automatically removes the entire folder 
        using shutil.rmtree() if fewer than 2 clean images remain.
"""

import os
import shutil
from facenet_pytorch import MTCNN
from PIL import Image
 
detector = MTCNN(keep_all=True, post_process=False)

def is_clean_image(image_path):
    try:
        img = Image.open(image_path).convert('RGB')
        boxes, _ = detector.detect(img)
        
        # Removes group shots and empty landscapes
        if boxes is not None and len(boxes) == 1:
            return True
        return False
    except Exception as e:
        return False

def clean_directory(dataset_dir):
    if not os.path.exists(dataset_dir):
        return

    for filename in os.listdir(dataset_dir):
        path = os.path.join(dataset_dir, filename)
        
        if os.path.isfile(path) and path.lower().endswith(('.png', '.jpg', '.jpeg', '.avif', '.webp')):
            if not is_clean_image(path):
                print(f"[-] Removing noisy/multi-face image: {filename}")
                try:
                    os.remove(path)
                except Exception as e:
                    pass

    # Need at least 2 photos per person for triplet loss to work
    remaining_images = [
        f for f in os.listdir(dataset_dir) 
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.avif', '.webp'))
    ]
    
    if len(remaining_images) < 2:
        print(f"[!] Folder '{dataset_dir}' has less than 2 valid images left. Deleting useless folder...")
        try:
            shutil.rmtree(dataset_dir)
        except Exception as e:
            print(f"[!] Could not delete folder {dataset_dir}: {e}")