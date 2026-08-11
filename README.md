# AI-Face-Similarity-Checker

An end-to-end computer vision pipeline built with **PyTorch**, **ResNet18**, and **MTCNN**. This project fine-tunes a deep feature extractor using **Triplet Margin Loss** to project human faces into a 128-dimensional metric hypersphere, enabling accurate facial verification, identity scoring, and $N \times M$ cross-comparison analysis.

---

## Key Features

* **Automated Face Scraping Engine:** Concurrent multi-threaded downloader (`scrape_faces.py`) pulling target face images directly via DuckDuckGo with duplicate URL detection.
* **MTCNN Face Alignment & Dataset Sanitation:** Automated pre-processing pipeline (`clean_dataset.py`) using **Multi-Task Cascaded Convolutional Networks (MTCNN)** to crop, align, and filter out group shots, empty backgrounds, and noisy samples.
* **Metric Learning Architecture:** Fine-tuned **ResNet18** backbone using **Triplet Margin Loss** ($A, P, N$ sampling) to minimize distance between matching identities while enforcing margin separation for distinct identities.
* **Mixed-Precision GPU Acceleration:** Optimized for high-throughput training using PyTorch **Automatic Mixed Precision (AMP)**, `AdamW` optimizer, and **Cosine Annealing** learning rate scheduling.
* **Interactive Multi-Mode CLI:** Comprehensive inference interface supporting:
  1. **1-on-1 Image Comparison:** Instant similarity scoring between two local face images.
  2. **Image vs. Identity Verification:** Score a target photo against a full folder of scraped images.
  3. **$N \times M$ Matrix Cross-Comparison:** Compute average and peak similarity matrices between two distinct identities.

---

## Project Architecture

```
AI-Face-Similarity-Checker/
├── bulk_scrape.py        # Automated sequential identity scraper
├── scrape_faces.py       # Concurrent web scraper & MTCNN face cropper
├── clean_dataset.py      # Automated dataset sanitation & single-face verifier
├── model.py              # ResNet18 128-d Feature Embedding Network
├── dataset.py            # On-disk PyTorch Triplet Sampler & DataLoader
├── train.py              # AMP-accelerated Triplet Margin Loss training loop
├── inference.py          # Interactive evaluation & similarity scoring CLI
├── visualize_grid.py     # Grid visualizer for scraped facial datasets
└──  requirements.txt      # Project dependencies
```

---

## Quickstart Guide

1. **Installation**
* Clone the repository and install dependencies:
```
  git clone https://github.com/S-M-5/AI-Face-Similarity-Checker
  cd AI-Face-Similarity-Checker
  pip install -r requirements.txt
```
---
2. **Scrape & Clean Data**
* Scrape target face datasets automatically using names.txt:
```
python scrape_faces.py
python clean_dataset.py
```
* Or if you want to scrape a large number of people make a file called names.txt in main folder, separate names using &, and run:
```
python bulk_scrape.py
python clean_dataset.py
```
---
3. **Train the Model (Skip this step to use the model I trained)**
* Fine-tune the ResNet18 feature extractor:
```
python train.py
```
---
4. **Run Interactive Inference**
* Launch the similarity checker CLI:
```
python inference.py
```
