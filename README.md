#  Player Re-Identification with YOLOv11 + CLIP ViT-B/16 + AURA

This project implements a modular player re-identification pipeline for a 15-second soccer video using **YOLOv11** for detection, **CLIP ViT-B/16** for appearance-based feature extraction, and a custom identity tracking algorithm called **AURA** (Anchor Unified Re-ID Algorithm).

---

##  What This Project Does

This pipeline:
- Assigns and preserves unique **player IDs** across a video
- Handles **occlusions**, **overlaps**, and **similar jerseys** via team clustering and anchor logic
- Uses **CLIP embeddings** to measure player similarity
- **Clusters jersey colors** using KMeans, dynamically filtering out the green field
- Creates an output video with overlayed bounding boxes showing:
  - Player ID
  - Team label
  - Frame number

All logic is modular and built from scratch without relying on prebuilt tracking packages.


##  How to Set Up and Run

> Built and tested on **Kaggle Kernels** using a **Tesla P100 GPU**.

###  Input Requirements
All input files are found at: 
Place the following in your `/input/` directory:
- `15sec_input_720p.mp4`: The input soccer video (as a dataset)
- YOLOv11 checkpoint (`best.pt`) (as a model)
- CLIP-ReID repo and ViT-B/16 weights (`weights_e8.pth`) (as datasets)
- Running this notebook on Kaggle avoids the need to change any file paths or environment settings — all input/output locations are already tailored for the Kaggle runtime.
> These paths are already hardcoded into the notebook using `/kaggle/input/...`

### ▶️ Running the Notebook

1. Open `clipaura.ipynb` in a Kaggle Notebook environment
2. Ensure GPU is enabled
3. Run all cells
