# Topic 1: U-Net Mask Generation & Vector Extraction

Before any geometric snapping or BIM modeling can occur, the raw raster floorplan image must be converted into a mathematical representation of walls. This process involves deep learning semantic segmentation and computer vision contour analysis.

## 1. Deep Learning Segmentation (U-Net)
The pipeline begins by analyzing the image using a **U-Net** architecture with a **ResNet-34 encoder**. 
- **Learning Point (Basics):** Semantic segmentation assigns a class label (e.g., "Wall", "Not Wall") to every single pixel in an image.
- **Learning Point (Advanced):** U-Net is an encoder-decoder network. The encoder extracts high-level spatial features (what an object is), while the decoder uses skip connections to recover the exact spatial resolution (where the object is).
- **Code Reference:** `pipeline.py` -> `load_model_logic()` loads the `best_cleaner_model_v3.pth` PyTorch model. `predict_tiled()` splits the image into `512x512` chunks and processes them using the CNN, stitching them back together using a Hann-window blending function to prevent visible seams.

## 2. Raster-to-Vector Extraction
Once the U-Net produces a clean binary mask of walls, we must convert these pixel regions into pure 1D line segments (vectors).
- **Learning Point (Basics):** A graph is a mathematical structure mapping relationships between nodes (points) and edges (lines connecting them).
- **Learning Point (Advanced):** We use **Skeletonization** (a morphological operation) to reduce thick pixel walls into 1-pixel-wide lines. We then extract pixels with exactly one neighbor (endpoints) or >2 neighbors (junctions) to define the graph nodes. 
- **RDP (Ramer-Douglas-Peucker) Algorithm:** We simplify jagged pixel paths into straight polylines by removing redundant intermediate points.
- **Code Reference:** `pipeline.py` -> `process_geometry()`. It invokes `extract_topology()` which performs `skimage.morphology.skeletonize`, and then `simplify_paths()` applies the RDP algorithm to collapse the vectors into straight architectural lines.

## External Resources & Further Reading
1. **U-Net Architecture:** [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
2. **Skeletonization Algorithms:** [Zhang-Suen thinning algorithm](https://en.wikipedia.org/wiki/Thinning_(morphology))
3. **Ramer-Douglas-Peucker Algorithm:** [RDP Polyline Simplification](https://en.wikipedia.org/wiki/Ramer%E2%80%93Douglas%E2%80%93Peucker_algorithm)
