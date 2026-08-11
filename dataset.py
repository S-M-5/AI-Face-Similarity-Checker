"""
Description:

Manages on-disk folder structure indexing and samples random image triplets for metric learning.

Class: TripletFaceDataset(Dataset)
    Role: A custom PyTorch Dataset that reads identity folders directly from disk and constructs Anchor-Positive-Negative image sets on the fly.
    Methods:
        1 - __init__(self, root_dir="data", transform=None): Scans the dataset directory, indexes valid folders, filters out identities with fewer than 2 photos,
        and indexes available classes.

        2 - __len__(self): Returns the total sum of images across all valid identity folders.

        3 - _get_image(self, img_path): Opens an image via PIL safely, handling unreadable/corrupted files by returning a blank black tensor fallback.

        4 - __getitem__(self, idx): Samples a random identity class to select an Anchor image and a distinct Positive image,
            then selects a random secondary identity class to pull a Negative image.
"""

import os
import random
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class TripletFaceDataset(Dataset):
    def __init__(self, root_dir="data", transform=None):

        self.root_dir = root_dir
        self.transform = transform
        self.label_to_images = {}
        
        print(f"[*] Scanning directory '{root_dir}' for identities... This might take a few seconds.")
        folders = [f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))]
        
        for folder in folders:
            folder_path = os.path.join(root_dir, folder)
    
            images = [
                os.path.join(folder_path, img) 
                for img in os.listdir(folder_path) 
                if img.lower().endswith(('.png', '.jpg', '.jpeg', '.avif', '.webp'))
            ]
            
            # Makes sure there's at least 2 photos per person
            if len(images) >= 2:
                self.label_to_images[folder] = images
                
        self.classes = list(self.label_to_images.keys())
        
        if len(self.classes) < 2:
            raise ValueError("Dataset error: You need at least 2 unique identities, with 2+ images each.")
            
        total_images = sum(len(imgs) for imgs in self.label_to_images.values())
        print(f"[+] Successfully indexed {len(self.classes)} valid identities ({total_images} total images) directly from disk.")

    def __len__(self):
        return sum(len(imgs) for imgs in self.label_to_images.values())

    def _get_image(self, img_path):
        try:
            return Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"[!] Warning: Could not read image {img_path}: {e}")
            # Turn bad image into a black image to not interfere with training
            return Image.new('RGB', (160, 160), color='black')

    def __getitem__(self, idx):
        anchor_cls = random.choice(self.classes)
        anchor_path, pos_path = random.sample(self.label_to_images[anchor_cls], 2)
        neg_cls = random.choice([c for c in self.classes if c != anchor_cls])
        neg_path = random.choice(self.label_to_images[neg_cls])
        
        anchor_img = self._get_image(anchor_path)
        pos_img = self._get_image(pos_path)
        neg_img = self._get_image(neg_path)
    
        if self.transform:
            anchor_img = self.transform(anchor_img)
            pos_img = self.transform(pos_img)
            neg_img = self.transform(neg_img)
            
        return anchor_img, pos_img, neg_img

face_transforms = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_transforms = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

eval_transforms = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

if __name__ == "__main__":
    print("Testing Direct Disk Dataset Loader...")
    dataset = TripletFaceDataset(root_dir="data", transform=face_transforms)
    a, p, n = dataset[0]
    print(f"Success! Pulled Triplet Tensor Shapes: {a.shape}, {p.shape}, {n.shape}")