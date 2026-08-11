"""
Description:

A automation script for scraping large lists of target names sequentially.

Script Functionality:
    - Reads names.txt and splits target identities separated by &.
    - Iterates through all names, invoking build_face_dataset() for each target.
    - Introduces a 5-second sleep delay between scraping iterations to respect search engine rate limits and prevent IP bans.

    ** names.txt was a file made to make gathering data quicker.
"""

import time
from scrape_faces import build_face_dataset

if __name__ == "__main__":
    print("=== Bulk 10,000 Name Scraper ===")
    
    with open("names.txt", "r", encoding="utf-8") as file:
        content = file.read()
        names = [name.strip() for name in content.split('&') if name.strip()]
        
    print(f"[*] Loaded {len(names)} names from file.")
    
    for i, target_name in enumerate(names):
        print(f"\n--- [{i+1}/{len(names)}] Processing: {target_name} ---")
        
        # Scrape 30 images per person
        build_face_dataset(target_name, max_images=30)
        
        print("[~] Sleeping for 5 seconds to avoid DuckDuckGo rate limits...")
        # avoid IP ban
        time.sleep(5)