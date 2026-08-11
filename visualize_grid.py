"""
Description:

A dataset inspection tool for visually checking scraped facial crops.

Function:
    plot_face_grid(folder_path, cols=5, max_images=40): Loads up to max_images from a target folder and renders them
    in an organized Matplotlib grid with title headers for quick visual auditing.
"""

import os
import math
import matplotlib.pyplot as plt
from PIL import Image

def plot_face_grid(folder_path, cols=5, max_images=40):
    if not os.path.exists(folder_path):
        print(f"[!] Folder '{folder_path}' does not exist.")
        return

    all_files = os.listdir(folder_path)
    image_files = sorted([f for f in all_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    image_files = image_files[:max_images]
    num_images = len(image_files)

    if num_images == 0:
        print(f"[!] No valid images found in '{folder_path}'.")
        return

    rows = math.ceil(num_images / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))

    if num_images == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    print(f"[*] Displaying {num_images} images in a {rows}x{cols} grid...")

    for i in range(len(axes)):
        if i < num_images:
            img_path = os.path.join(folder_path, image_files[i])
            try:
                img = Image.open(img_path).convert('RGB')
                axes[i].imshow(img)
                axes[i].set_title(image_files[i], fontsize=8)
            except Exception as e:
                axes[i].text(0.5, 0.5, 'Error Loading', ha='center', va='center')
        
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()