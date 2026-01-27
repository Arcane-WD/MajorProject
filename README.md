# 🏗️ Scan-to-BIM: AI-Driven Floorplan to 3D Reconstruction

This project implements an **end-to-end Scan-to-BIM pipeline** that converts a **2D architectural floorplan image** into a **metric-scaled, navigable 3D building model (GLB)** using deep learning and computational geometry.

The system combines **CNN-based perception**, **graph-based drafting**, and **procedural 3D construction** into a single reproducible pipeline with an interactive web interface.

---

## 🔥 What This Project Does

**Input**
A raster image of a building floorplan (PNG / JPG)

**Output**
A **true-scale 3D building** with:

* Walls
* Doors (with headers / lintels)
* Floor slab
* Correct physical dimensions
* Exportable as `.glb`
* Viewable interactively in browser

This is not a mesh generator — it is a **geometry-aware reconstruction engine**.

---

## 🧠 System Architecture

```
Floorplan Image
      │
      ▼
[ Phase 1 ]  CNN (UNet) — Wall Segmentation
      │
      ▼
[ Phase 2 ]  Skeleton → Graph → Vectorized Walls
      │
      ▼
[ Phase 3 ]  Procedural BIM-Style Construction
      │
      ▼
     GLB 3D Building
```

Each phase is modular and independently testable.

---

## 🧩 Phase Breakdown

### Phase 1 — Perception (Deep Learning)

* Model: **UNet (ResNet-34 encoder)**
* Task: Pixel-wise wall segmentation
* Trained on: ~4,200 floorplan images
* Output: Probability mask of wall locations

Features:

* ImageNet-normalized inference
* Aspect-ratio preserving resize with padding
* GPU-accelerated PyTorch inference

---

### Phase 2 — Drafting (Geometry Extraction)

Converts CNN masks into **CAD-like vectors**

Pipeline:

1. Threshold → Binary mask
2. Morphological gap closing
3. Skeletonization (1-pixel wide walls)
4. Graph construction (4-connectivity)
5. Junction detection
6. Path tracing
7. Ramer-Douglas-Peucker (RDP) simplification
8. Deduplication & pruning

Output:

* Clean orthogonal wall segments (pixel coordinates)

This phase turns **images into geometry**.

---

### Phase 3 — Construction (3D BIM Engine)

Procedural architecture engine built with **Trimesh**

Features:

* Pixel → Meter scaling (true-scale)
* Walls as solid volumes
* Floor slab
* Door gap detection
* Header (lintel) generation above doors
* Collinearity-aware door detection
* Watertight 3D geometry

Output:

* Exportable `.glb` building model

---

## 🖥️ Web Interface

The project includes a **Streamlit web app** that lets users:

* Upload a floorplan image
* Run the full AI → BIM pipeline
* View the generated 3D model in browser
* Rotate / zoom / inspect the building
* Download the `.glb` file

The viewer is powered by **Google `<model-viewer>`** embedded via Base64, so no backend file server is needed.

---

## 📦 Repository Contents

```
app.py               → Streamlit web interface
pipeline.py     → Full Scan-to-BIM engine
model_links.txt      → Google Drive link to trained UNet
sample io/              → Demo inputs & outputs
```

The model file (`best_cleaner_model.pth`) is stored externally due to GitHub size limits.

---

## ✅ Current System Capabilities

| Feature                   | Status |
| ------------------------- | ------ |
| Wall segmentation (CNN)   | ✅      |
| Skeletonization           | ✅      |
| Graph-based vectorization | ✅      |
| RDP simplification        | ✅      |
| Door gap detection        | ✅      |
| Door headers (lintels)    | ✅      |
| Metric scaling            | ✅      |
| Floor slab                | ✅      |
| 3D export (GLB)           | ✅      |
| Web UI                    | ✅      |

---

## ⚠️ Known Limitations (V1)

* Uses 512×512 CNN inference → some wall wobble
* Door detection can fail on low-resolution masks
* Windows and room semantics not yet modeled
* No multi-floor support

These are solved in **Phase 4–6**.

---

## 🔜 Roadmap

### Phase 4 — High-Fidelity Inference

Tiled CNN inference on high-resolution images to produce CAD-grade masks.

### Phase 5 — Semantic BIM

Room detection, doors, windows, room labels.

### Phase 6 — Parametric Architecture

Editable wall thickness, door size, IFC/Revit export.

### Phase 7 — Multi-Floor & Scale Calibration

---

## 🏆 Why This Project Is Different

Most “floorplan to 3D” projects output **meshes**.
This system outputs **architecture**.

It explicitly models:

* topology
* geometry
* scale
* doors
* walls
* floors

That is what makes it **BIM-grade**, not a visualization toy.

---

## 🚀 Status

**V1.1 — Hardened Geometry Engine + Web UI**

The system is fully functional and produces usable 3D buildings from raw images.

Further phases will refine accuracy and semantics — not reinvent the pipeline.

