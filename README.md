# Scan-to-BIM

**AI-Driven 2D Floorplan to BIM-Grade 3D Reconstruction**

This project implements a **research-oriented Scan-to-BIM pipeline** that converts a **2D architectural floorplan image** into a **metric-scaled, editable 3D building model (GLB)** using deep learning, computational geometry, and raster-to-vector reconstruction.

Unlike mesh-based visualizers, the system reconstructs **architectural structure**—walls, doors, topology, and scale—suitable for BIM-style downstream use.

---

## Project Objective

### Input

A raster floorplan image (PNG / JPG)

### Output

A **metric-scale 3D building model** containing:

* Walls represented as solids
* Door openings with headers (lintels)
* Floor slab
* Consistent pixel-to-meter scaling
* Exportable `.glb` model
* Interactive browser-based visualization

The system focuses on **geometric reconstruction**, not surface visualization.

---

## System Architecture

```
Floorplan Image
      │
      ▼
[ Phase 1 ]  CNN-Based Wall Probability Estimation
      │
      ▼
[ Phase 4 ]  High-Resolution Mask Generation
      │
      ▼
[ Phase 5 ]  Raster-to-Vector Geometry Extraction
      │
      ▼
[ Phase 3 ]  Procedural BIM-Style Construction
      │
      ▼
     GLB 3D Building Model
```

Each phase is modular and independently extensible.

---

## Phase Overview

---

### Phase 1 — Perception (Deep Learning)

**Status:** Implemented

* Model: UNet with ResNet-34 encoder
* Task: Pixel-wise wall probability estimation
* Output: Floating-point probability map

Key characteristics:

* ImageNet-normalized inference
* Aspect-ratio–preserving resizing
* GPU-accelerated PyTorch pipeline

---

### Phase 4A — High-Resolution Tiled Inference

**Status:** Implemented

Addresses the resolution mismatch between fixed-input CNNs and large floorplans.

* Input images are divided into overlapping 512×512 tiles
* Each tile is independently processed by the CNN
* Outputs are merged using Hann-window weighted blending

This preserves wall continuity and global geometry while maintaining high spatial resolution.

This approach is conceptually aligned with SAHI-style inference.

---

### Phase 4B — Mask Refinement

**Status:** Implemented

Post-processing of CNN probability maps to ensure structural integrity before vectorization.

Operations include:

* Hard thresholding
* Connected-component filtering for noise removal
* Morphological closing to bridge small gaps
* Morphological opening for edge smoothing

The output is a clean, contiguous wall mask suitable for geometric processing.

---

### Phase 5A — Hybrid Raster-to-Vector Conversion

**Status:** Implemented

This phase performs the core Scan-to-BIM transformation by combining:

* Skeleton-based topology extraction
* Pixel-region–based geometric fitting

Pipeline steps:

1. Skeletonization of the refined mask
2. Conversion of the skeleton into a graph
3. Tracing of wall paths between junctions
4. Segmentation at geometric bends using RDP
5. Extraction of wall pixel regions around each segment
6. Least-squares (PCA) line fitting on pixel clouds
7. Generation of CAD-grade wall axes

This produces straight, metric-accurate wall vectors while preserving connectivity.

---

### Phase 5B — Junction and Topology Optimization

**Status:** Implemented

Enhancements include:

* Vertex snapping and corner closure
* Manhattan-world (orthogonality) enforcement
* Noise-gap healing and wall truncation bridging

Objective: to produce a topologically consistent floorplan suitable for semantic reasoning.

---

### Phase 2C — Semantic Object Detection & Structural Correction

**Status:** Implemented

Incorporates YOLOv8 object detection to identify architectural features (doors, sliding doors) and natively carve topological gaps into the vectorized walls based on advanced probabilistic geometric slicing heuristics.

---

### Phase 3 — BIM-Style 3D Construction

**Status:** Implemented

Using Trimesh, vector geometry is converted into 3D architectural solids:

* Pixel-to-meter scaling
* Wall extrusion with thickness and height
* Door gap detection
* Header (lintel) generation
* Floor slab construction
* Watertight GLB export

---

### Phase 6 — Gamified 3D Navigation Layer (Babylon.js)

**Status:** Implemented

Replaces the passive model viewer with a First-Person fully interactive spatial experience natively injected into Streamlit:
* **Physics & Collision:** Bounding Centroid calculations force the Universal Camera to spawn cleanly inside the geometry natively.
* **Canvas Minimaps:** Static HTML canvas rendering the 2D bounding walls with a live telemetry player tracker.
* **Raycast Interaction:** Click-and-drag mouse look with a 2.5-meter Raycast establishing physical $90^\circ$ dynamic door swings.

---

### Phase 7 — Parametric BIM Extensions & Multi-Floor

**Status:** Planned

* Editable wall thickness and door dimensions
* IFC / Revit-compatible export
* Stair detection and floor stacking
* Absolute scale calibration

---

## Academic Context & Constrained Compute Constraints

This engine achieves state-of-the-art heuristic 3D reconstruction **exclusively utilizing free-tier constrained resources (1x Kaggle P100/T4, 30 hrs/week)**. 

Compared to contemporary "Floorplan-to-3D" repositories:
1. **FloorPlanTo3D-unityClient (Custom Mask R-CNN)** utilizes heavy multi-day training loops on Resnet101 backbones to extract geometric walls. Instead, we shifted entirely to a fast morphology-cleaned **U-Net** topology generator.
2. **3DPlanNet & DeepFloorplan (Hybrid Approaches)** utilize similar object detection logic, however typically render offline using heavy Desktop clients. We achieve real-time topological slicing *purely* by math projections mapped to **YOLOv8 Medium**, keeping total VRAM utilization dramatically low (~12GB max).
3. **Babylon.js & Streamlit vs. Unity/Unreal Engine:** Unlike Unity-client implementations, this project runs the *entire* spatial simulation seamlessly via sandboxed HTML `iframes` requiring exactly 0 external dependencies besides standard pip modules.

By balancing Deep Learning (U-Net & YOLO) strictly for *prediction*, and Computational Geometry (Raster-to-Vector topology gaps) for *reconstruction*, this engine mathematically exceeds the hardware limits of local consumer graphics cards while maintaining high precision.

---

## Web Interface

A Streamlit-based interface provides:

* Floorplan upload
* Inference mode selection (Fast / High-Fidelity)
* End-to-end Scan-to-BIM execution
* Interactive 3D visualization
* GLB export

The viewer is implemented using Google’s `<model-viewer>` and Base64-embedded GLB rendering.

---

## Repository Structure

```
app.py        → Web interface
pipeline.py   → Scan-to-BIM processing pipeline
model_links.txt
sample_io/
```

---

## Current Capabilities

| Capability                      | Status      |
| ------------------------------- | ----------- |
| CNN-based wall detection        | Implemented |
| High-resolution tiled inference | Implemented |
| Mask refinement                 | Implemented |
| Skeleton-based topology         | Implemented |
| Pixel-cloud line fitting        | Implemented |
| CAD-grade wall vectors          | Implemented |
| Phase 5B (Geometric Cleanup)    | Implemented |
| Supervised YOLOv8 Training      | Implemented (mAP50=0.80) |
| Phase 2C (Wall-Gap Correction)  | Implemented |
| Metric scaling                  | Implemented |
| BIM-style 3D model              | Implemented |
| Web-based visualization         | Implemented |

---

## Known Limitations

* Thin or ambiguous walls may be missed by the U-Net.
* Intra-class YOLOv8 confusion (single vs. double vs. sliding doors), mitigated by treating all door categories interchangeably for structural logic.
* **Furniture Orientation:** Furniture orientation relies on snap-alignment to the nearest wall, which could be inaccurate for center-room furniture.
* Single-floor support only.

---

## Project Scope and Distinction

Most systems generate **surface meshes**.
This system reconstructs **architectural geometry**.

It explicitly models topology, geometry, and scale to generate a true BIM-oriented pipeline.

---

## Current Status

**Version 2.0 — Structural Scan-to-BIM Engine**

* **Phase 1 (Geometry Execution):** U-Net Inference, Trimesh routing, and Phase 5B Junction Optimization are fully implemented.
* **Phase 2 (Architectural Detection):** A custom YOLOv8 Object Detection model has been happily trained on Kaggle (mAP50=0.80) over 15k FloorPlanCAD samples.
* **Phase 2C (Structural Correlation):** Completely implemented! High-fidelity YOLO detections retroactively project probabilistically onto Phase 1 maps, natively slicing intersections for real physically extracted door openings and placing dynamically sized furniture.

---

## Evaluation Metrics

The core Phase 1 U-Net perception model was evaluated on a held-out dataset of 50 samples:

| Model | IoU | Dice | Precision | Recall | F1 | Accuracy |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| v1 (10ep) | 0.8224 | 0.9005 | 0.8957 | 0.9080 | 0.9005 | 0.9817 |
| v2 (+30ep) | 0.9333 | 0.9653 | 0.9716 | 0.9593 | 0.9653 | 0.9936 |
| **v3 (+30ep)** | **0.9613** | **0.9801** | **0.9824** | **0.9781** | **0.9801** | **0.9964** |
| v4 (aug+val) | 0.9190 | 0.9576 | 0.9415 | 0.9745 | 0.9576 | 0.9921 |

The active repository currently uses the **v3 weights**.
