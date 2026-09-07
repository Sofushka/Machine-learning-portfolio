# Satellite Image Keypoint Matching

This repository contains an end-to-end Deep Learning pipeline for detecting keypoint correspondences between satellite images (Sentinel-2) taken at different dates/seasons.

* **Original Dataset:** [Kaggle - Deforestation in Ukraine](https://www.kaggle.com/datasets/isaienkov/deforestation-in-ukraine)
* **Processed Tiles Dataset:** [Google Drive - Processed Pairs 512x512](https://drive.google.com/file/d/176wtJIsB1RIv2Axe-f2nplRXoebqGTGy/view?usp=sharing)
* **Pretrained Weights:** Integrated via Kornia LoFTR Outdoor Weights (`kornia.feature.LoFTR`)

---

## Architecture & Approach

Standard classical methods (like SIFT, ORB) fail on multi-temporal satellite imagery due to significant seasonal variations (snow coverage, vegetation growth, illumination shifts).

To address this:
1. **Model:** **LoFTR (Local Feature TRansformer)** — a detector-free SOTA model that uses self and cross-attention mechanisms in Transformers to extract dense keypoints even in low-texture or seasonal-change regions.
2. **Geometric Verification:** **USAC_MAGSAC (RANSAC)** — filters spurious matches by calculating the homography matrix, separating correspondences into **Inliers (Green)** and **Outliers (Red)**.

---

## Repository Structure

```text
task2/
├── dataset_creation.ipynb              # Download & tiling pipeline (512x512 patches)
├── demo.ipynb                          # Demo notebook with keypoint match visualization
├── inference.py                        # Core SatelliteImageMatcher class (LoFTR + MAGSAC)
├── train.py                            # Automated benchmark evaluator across image pairs
├── requirements.txt                    # Task dependencies
├── Report with potential improvements.pdf # Performance analysis and future ideas
└── README.md                           # Task documentation
