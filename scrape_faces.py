"""
Description:

Automates image collection from web search engines, filtering and cropping faces concurrently.

Functions:
    1 - fetch_image_data(url): Performs an HTTP GET request disguised with custom user-agent headers to download raw image bytes into PIL memory.
    
    2 - build_face_dataset(person_name, max_images=30): Queries DuckDuckGo for thumbnail URLs, processes downloads concurrently 
        using ThreadPoolExecutor(max_workers=16), detects faces with MTCNN, crops valid face regions directly to disk (data/person_name/),
        and logs processed URLs to prevent duplicate downloads.
"""
import os
import requests
import concurrent.futures
from ddgs import DDGS
from facenet_pytorch import MTCNN
from PIL import Image
from io import BytesIO

from visualize_grid import plot_face_grid


def fetch_image_data(url):
    # Disguise the Python script as a standard Google Chrome web browser
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    try:
        response = requests.get(url, headers=headers, timeout=5)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert('RGB')
        return url, img
    except Exception:
        return url, None

def build_face_dataset(person_name, max_images=30):
    folder_name = person_name.lower().replace(" ", "_")
    final_dir = f"data/{folder_name}"
    os.makedirs(final_dir, exist_ok=True)
    
    # Prevents re-downloading images
    history_file = os.path.join(final_dir, "downloaded_urls.txt")
    seen_urls = set()
    if os.path.exists(history_file):
        with open(history_file, "r", encoding="utf-8") as f:
            seen_urls = set(line.strip() for line in f)
            
    valid_count = 0
    while os.path.exists(os.path.join(final_dir, f"{folder_name}_{valid_count}.jpg")):
        valid_count += 1
        
    print(f"[*] Searching DuckDuckGo for: {person_name} (Found {len(seen_urls)} previously processed URLs)")
    
    urls_to_process = []
    try:
        with DDGS() as ddgs:
            search_results = ddgs.images(person_name, max_results=max_images)
            
            for res in search_results:
                img_url = res.get("thumbnail") or res.get("image")
                
                if img_url and img_url not in seen_urls:
                    urls_to_process.append(img_url)
                
                if len(urls_to_process) >= max_images * 10:
                    break
    except Exception as e:
        print(f"[!] DuckDuckGo Search failed: {e}")
        return final_dir, valid_count

    detector = MTCNN(image_size=160, margin=10, select_largest=True, post_process=False)
    added_count = 0
    
    print(f"[*] Downloading and processing {len(urls_to_process)} URLs concurrently (10 workers)...")
    
    with open(history_file, "a", encoding="utf-8") as history:
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
           
            future_to_url = {executor.submit(fetch_image_data, url): url for url in urls_to_process}
            
            for future in concurrent.futures.as_completed(future_to_url):
                if added_count >= max_images:
                    break  
                
                url, img = future.result()
                if img is None:
                    continue 
                
                try:
                    boxes, _ = detector.detect(img)
                    
                    if boxes is not None and len(boxes) >= 1:
                        save_path = os.path.join(final_dir, f"{folder_name}_{valid_count}.jpg")
                        detector(img, save_path=save_path)
                        
                        valid_count += 1
                        added_count += 1

                        seen_urls.add(url)
                        history.write(url + "\n")
                except Exception:
                    continue

    print(f"[+] Successfully verified and added {added_count} new face images to {final_dir}/")
    return final_dir, valid_count

if __name__ == "__main__":
    print("=== Automated Face Dataset Generator ===")
    
    user_input = input("Enter names separated by '&' (e.g., Messi & Ronaldo): ")
    names = [name.strip() for name in user_input.split('&') if name.strip()]
    
    try:
        max_limit = int(input("Max *new* images per person to scrape this run (e.g., 40): "))
    except ValueError:
        print("Invalid number. Defaulting to 40.")
        max_limit = 40
        
    for target_name in names:
        print(f"\n--- Processing Pipeline for: {target_name} ---")
        
        final_dir, total_count = build_face_dataset(target_name, max_images=max_limit)
        
        if total_count > 0:
            plot_face_grid(final_dir, max_images=total_count)
            
    print("\n[+] All requested scraping tasks completed!")